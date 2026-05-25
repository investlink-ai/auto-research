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

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import click
import httpx
import pyarrow.parquet as pq

from auto_research._io import atomic_write_text
from auto_research.extract.workers.s_filings import extract_s_filing
from auto_research.ingest.edgar import EdgarConfigError, fetch_filings_for_cik
from auto_research.telemetry import try_init_telemetry

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


def _normalize_cik(raw: str) -> str:
    """Strip whitespace and zero-pad to 10 digits (canonical EDGAR form).

    The manifest stores entity_id in the padded form (edgar.py calls
    `_pad_cik` at write time). Accepting unpadded or whitespace-padded
    input at the CLI boundary keeps `ingest edgar --cik X` and
    `extract s-filings --cik X` symmetric for the same operator input.
    """
    cleaned = raw.strip()
    if not cleaned.isdigit():
        raise click.UsageError(
            f"--cik must be digits only (e.g., 0001045810 or 1045810); got {raw!r}"
        )
    return cleaned.zfill(10)


def _feast_repo_or_exit() -> Path:
    """Resolve `_FEAST_REPO_DIR` against CWD; raise a clean UsageError if missing.

    `subprocess.run(..., cwd=Path)` raises an unhelpful FileNotFoundError if
    the directory is absent. Surface the prerequisite at the CLI boundary so
    the operator gets a one-line remediation instead of a traceback.
    """
    repo = _FEAST_REPO_DIR.resolve()
    if not repo.is_dir():
        raise click.UsageError(
            f"feast_repo/ not found at {repo}. Run from the project root."
        )
    return repo


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
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=_DEFAULT_RAW_ROOT,
    show_default=True,
    help="Root directory for raw bytes.",
)
@click.option(
    "--manifest-path",
    type=click.Path(dir_okay=False, path_type=Path, resolve_path=True),
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
    try_init_telemetry()
    cik = _normalize_cik(cik)
    parsed_forms = tuple(f.strip() for f in form_types.split(",") if f.strip())
    if not parsed_forms:
        raise click.UsageError(
            "--form-types must contain at least one non-empty form type"
        )
    try:
        results = fetch_filings_for_cik(
            cik,
            form_types=parsed_forms,
            raw_root=raw_root,
            manifest_path=manifest_path,
        )
    except EdgarConfigError as exc:
        raise click.UsageError(str(exc)) from exc
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
    type=click.Path(dir_okay=False, exists=True, path_type=Path, resolve_path=True),
    default=_DEFAULT_MANIFEST,
    show_default=True,
    help="Manifest Parquet file.",
)
@click.option(
    "--out-root",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=_DEFAULT_EXTRACTED_ROOT,
    show_default=True,
    help="Where to persist SFilingOutput JSON (worker-keyed subdir).",
)
@click.option(
    "--quarantine-root",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=None,
    help=(
        "Override the worker's default quarantine root "
        "(forwarded to extract_s_filing). Defaults to data/quarantine/."
    ),
)
def extract_s_filings(
    cik: str,
    manifest_path: Path,
    out_root: Path,
    quarantine_root: Path | None,
) -> None:
    try_init_telemetry()
    cik = _normalize_cik(cik)
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
    skipped = 0
    failed = 0
    for row in candidates:
        raw_path = Path(row["path"])
        try:
            # `errors="replace"` is the canonical decode policy for the
            # worker's INV-6 cache key (sha256 over raw_doc.encode()). A
            # future caller using read_bytes() or a different encoding will
            # compute a divergent key for the same content and pollute the
            # cache. Add another caller? Route it through the same policy or
            # refactor extract_s_filing to take bytes.
            raw_doc = raw_path.read_text(errors="replace")
        except OSError as exc:
            click.echo(f"warn: skipping {row['doc_id']} (file unreadable): {exc}", err=True)
            skipped += 1
            continue
        try:
            result = extract_s_filing(
                raw_doc=raw_doc,
                doc_id=row["doc_id"],
                quarantine_root=quarantine_root,
            )
        except Exception as exc:
            click.echo(
                f"warn: extraction failed for {row['doc_id']}: "
                f"{exc.__class__.__name__}: {exc}",
                err=True,
            )
            failed += 1
            continue
        if result is None:
            quarantined += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            out_dir / f"{row['doc_id']}.json",
            result.model_dump_json(indent=2),
        )
        persisted += 1
    click.echo(
        f"extract s-filings: cik={cik} candidates={len(candidates)} "
        f"persisted={persisted} quarantined={quarantined} "
        f"skipped={skipped} failed={failed}"
    )


@cli.group(name="feast", help="Wrap the Feast CLI against feast_repo/.")
def feast_group() -> None: ...


@feast_group.command("apply", help="Run `feast apply` in feast_repo/.")
def feast_apply() -> None:
    try_init_telemetry()
    proc = subprocess.run(["feast", "apply"], cwd=_feast_repo_or_exit(), check=False)
    raise SystemExit(proc.returncode)


@feast_group.command(
    "materialize",
    help="Run `feast materialize START END` in feast_repo/. Dates are ISO 8601.",
)
@click.option(
    "--start",
    required=True,
    help="Inclusive ISO 8601 datetime (e.g., 2024-01-01T00:00:00).",
)
@click.option(
    "--end",
    required=True,
    help="Inclusive ISO 8601 datetime (e.g., 2024-01-31T00:00:00).",
)
def feast_materialize(start: str, end: str) -> None:
    try_init_telemetry()
    proc = subprocess.run(
        ["feast", "materialize", start, end],
        cwd=_feast_repo_or_exit(),
        check=False,
    )
    raise SystemExit(proc.returncode)


CheckStatus = Literal["ok", "warn", "error"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str


def _mask_url_credentials(url: str) -> str:
    """Return `url` with any embedded basic-auth credentials replaced by `***`.

    Defends INV-7: a LANGFUSE_HOST like `http://user:pass@host/` must not
    print credentials to stdout via the status command's detail line.
    """
    parts = urlsplit(url)
    if not (parts.username or parts.password):
        return url
    host = parts.hostname or ""
    netloc = f"***@{host}:{parts.port}" if parts.port else f"***@{host}"
    return urlunsplit(parts._replace(netloc=netloc))


def _check_langfuse() -> CheckResult:
    """Probe Langfuse: env presence + HTTP GET to /api/public/health."""
    host = (
        os.environ.get("LANGFUSE_HOST", "http://localhost:3000").strip().rstrip("/")
    )
    otlp = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    missing = [
        name
        for name, value in (
            ("OTEL_EXPORTER_OTLP_ENDPOINT", otlp),
            ("LANGFUSE_PUBLIC_KEY", pk),
            ("LANGFUSE_SECRET_KEY", sk),
        )
        if not value
    ]
    if missing:
        return CheckResult(
            "langfuse",
            "warn",
            f"missing env: {', '.join(missing)} (copy .env.example to .env)",
        )
    display_host = _mask_url_credentials(host)
    try:
        resp = httpx.get(f"{host}/api/public/health", timeout=2.0)
        if resp.status_code == 200:
            return CheckResult("langfuse", "ok", display_host)
        return CheckResult(
            "langfuse",
            "error",
            f"{display_host} returned HTTP {resp.status_code}",
        )
    except httpx.HTTPError as exc:
        return CheckResult(
            "langfuse", "error", f"{display_host}: {exc.__class__.__name__}"
        )


def _check_mlflow() -> CheckResult:
    """Report the configured tracking URI; warn if file-backend dir missing.

    Both the lazy import and the URI-resolution call are guarded so a broken
    experiment module surfaces as `[error] mlflow ...` in the status output,
    not as an unhandled exception that crashes the whole list comprehension
    before any line prints.
    """
    try:
        from auto_research.experiment import configured_tracking_uri
    except Exception as exc:
        return CheckResult("mlflow", "error", f"import failed: {exc}")
    try:
        uri = configured_tracking_uri()
    except Exception as exc:
        return CheckResult("mlflow", "error", f"URI resolution failed: {exc}")
    if uri.startswith("file:"):
        path = Path(uri.removeprefix("file://").removeprefix("file:"))
        if path.exists():
            return CheckResult("mlflow", "ok", uri)
        return CheckResult("mlflow", "warn", f"{uri} (directory not yet created)")
    return CheckResult("mlflow", "ok", uri)


def _check_feast() -> CheckResult:
    """Load the Feast registry from ./feast_repo. Warn if not yet applied."""
    repo = Path("feast_repo")
    if not repo.exists():
        return CheckResult("feast", "warn", "feast_repo/ not found in CWD")
    registry_db = repo / "data" / "registry.db"
    if not registry_db.exists():
        return CheckResult(
            "feast",
            "warn",
            "feast_repo/ present but registry.db missing - run `auto-research feast apply`",
        )
    try:
        from feast import FeatureStore

        store = FeatureStore(repo_path=str(repo))
        views = store.list_feature_views()
        return CheckResult("feast", "ok", f"{len(views)} feature_view(s)")
    except Exception as exc:
        return CheckResult("feast", "error", f"registry load failed: {exc}")


_STATUS_SYMBOL = {"ok": "ok", "warn": "warn", "error": "error"}


@cli.command(help="Print Langfuse / MLflow / Feast registry health.")
def status() -> None:
    checks = [_check_langfuse(), _check_mlflow(), _check_feast()]
    for check in checks:
        click.echo(
            f"[{_STATUS_SYMBOL[check.status]:<4}] {check.name:<10} {check.detail}"
        )
    if any(c.status == "error" for c in checks):
        raise SystemExit(1)


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
