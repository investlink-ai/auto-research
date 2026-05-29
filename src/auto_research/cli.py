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
from auto_research.cli_eval import eval_group
from auto_research.extract.embeddings import _PER_CORPUS_STORE
from auto_research.extract.materialization import (
    ActiveMaterialization,
    append_promotion_history,
    list_materializations,
    now_utc_iso,
    read_active_materialization,
    read_promotion_history,
    split_versioned_table_name,
    versioned_table_name,
    write_active_materialization,
)
from auto_research.extract.workers.s_filings import extract_s_filing
from auto_research.ingest.edgar import EdgarConfigError, fetch_filings_for_cik
from auto_research.telemetry import try_init_telemetry

_DEFAULT_RAG_ROOT = Path("data/rag")

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


def _resolve_reembed_targets(
    rag_root: Path,
    *,
    doc_id: str | None,
    corpus: bool,
    all_: bool,
) -> tuple[list[str], bool]:
    """Translate the mutually-exclusive CLI flags into a per-doc list +
    a corpus flag.

    `--doc-id X` → (["X"], False); `--corpus` → ([], True); `--all` →
    (every `<rag_root>/*.lance` directory minus `_corpus_narrative`,
    False — corpus rows are propagated by per-doc reembed via vector-
    copy, so --all does NOT separately re-encode the corpus).

    Raises `click.UsageError` if zero or multiple targets were passed,
    if `--doc-id` is an empty string, or if `--doc-id` is the reserved
    corpus-store name (review findings #9 and #15).
    """
    # Treat `doc_id` presence by identity rather than truthiness so
    # `--doc-id ""` raises a meaningful error rather than being silently
    # dropped as 'not specified'.
    chosen: list[str] = []
    if doc_id is not None:
        chosen.append("--doc-id")
    if corpus:
        chosen.append("--corpus")
    if all_:
        chosen.append("--all")
    if len(chosen) != 1:
        raise click.UsageError(
            f"exactly one of --doc-id, --corpus, --all is required; got "
            f"{chosen if chosen else 'none'}"
        )
    if doc_id is not None:
        if not doc_id:
            raise click.UsageError("--doc-id must be a non-empty string")
        if doc_id == _PER_CORPUS_STORE:
            raise click.UsageError(
                f"--doc-id={doc_id!r} is the reserved corpus-narrative "
                "table name. Use --corpus to re-embed it explicitly."
            )
        return ([doc_id], False)
    if corpus:
        return ([], True)
    # --all: enumerate per-doc tables present at the ACTIVE materialization
    # version. Tables on disk live in `{base}__{materialization_version}.lance/`
    # directories under `rag_root`. The corpus base is filtered out — it
    # gets propagated by vector-copy during per-doc reembed and so is not
    # listed as an --all target (re-encoding it separately would double-
    # spend on Voyage and risk batch-boundary nondeterminism between
    # per-doc and corpus vectors).
    #
    # Without an active pointer (fresh install or pre-migration legacy
    # layout) the CLI has nothing to enumerate; raise a UsageError so the
    # operator hits a one-line remediation pointing at the migration
    # script or `embed` to build the initial materialization.
    if not rag_root.exists():
        return ([], False)
    active = read_active_materialization(rag_root)
    if active is None:
        raise click.UsageError(
            f"no active materialization under {rag_root}; --all has nothing "
            "to enumerate. Build an initial materialization via the embed "
            "path, then promote it before re-running --all."
        )
    per_doc: list[str] = []
    for entry in sorted(rag_root.iterdir()):
        if not (entry.is_dir() and entry.suffix == ".lance"):
            continue
        split = split_versioned_table_name(entry.stem)
        if split is None:
            continue
        base, version = split
        if version != active.version:
            continue
        if base == _PER_CORPUS_STORE:
            continue
        per_doc.append(base)
    return (per_doc, False)


@extract.command(
    "reembed",
    help=(
        "Re-encode already-embedded LanceDB tables against a new embed model "
        "without re-running contextual chunking. Encoder-only path."
    ),
)
@click.option(
    "--backend",
    type=click.Choice(["voyage", "bge", "qwen3-mlx"]),
    default="qwen3-mlx",
    show_default=True,
    help=(
        "Embedding backend. Defaults to qwen3-mlx (in-process, $0 marginal "
        "cost). EMBEDDING_BACKEND env var is NOT consulted — selection is "
        "explicit on the reembed surface."
    ),
)
@click.option("--doc-id", default=None, help="Single per-doc LanceDB table to re-embed.")
@click.option(
    "--corpus", is_flag=True, default=False, help="Re-embed the _corpus_narrative table only.",
)
@click.option(
    "--all",
    "all_",
    is_flag=True,
    default=False,
    help="Re-embed every per-doc table plus the corpus narrative table.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report expected token volume + Voyage USD cost; perform no encoder calls.",
)
@click.option(
    "--rag-root",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=_DEFAULT_RAG_ROOT,
    show_default=True,
    help="LanceDB root directory containing per-doc and corpus tables.",
)
@click.option(
    "--voyage-model",
    default=None,
    help="Voyage model id (only valid with --backend voyage).",
)
@click.option(
    "--mlx-qwen3-model",
    default=None,
    help="Qwen3-MLX model id (only valid with --backend qwen3-mlx).",
)
def extract_reembed(
    backend: str,
    doc_id: str | None,
    corpus: bool,
    all_: bool,
    dry_run: bool,
    rag_root: Path,
    voyage_model: str | None,
    mlx_qwen3_model: str | None,
) -> None:
    try_init_telemetry()
    from typing import Literal, cast

    from auto_research.extract.embeddings import (
        _MODEL_DIM,
        _VOYAGE_USD_PER_MTOK,
        EmbeddingAdapter,
        embed_model_version,
    )

    per_doc_targets, include_corpus = _resolve_reembed_targets(
        rag_root, doc_id=doc_id, corpus=corpus, all_=all_,
    )

    # rag_root.exists() is validated up-front for BOTH dry-run and the
    # live path. Pre-fix, only dry-run checked, so a typo'd live
    # invocation hit a misleading 'table not found' from lancedb after
    # silently creating the directory (review finding #7).
    if not rag_root.exists():
        raise click.UsageError(f"--rag-root {rag_root} does not exist")

    # Construct the adapter once. Adapter init validates that the
    # backend/model kwargs are coherent (e.g., voyage_model with
    # backend=bge is rejected by the adapter, not silently dropped here).
    backend_lit = cast(Literal["voyage", "bge", "qwen3-mlx"], backend)
    adapter = EmbeddingAdapter(
        backend=backend_lit,
        rag_root=rag_root,
        voyage_model=voyage_model,
        mlx_qwen3_model=mlx_qwen3_model,
    )

    version_token = embed_model_version(backend_lit, adapter.model)
    new_dim = _MODEL_DIM[adapter.model]

    if dry_run:
        # Token + USD estimate WITHOUT touching the encoder. Read the
        # `text` column from every target table and sum tokens via the
        # chunker's `count_tokens` (cl100k_base) — same tokenizer used
        # for chunk-budget arithmetic upstream, so the numbers are
        # comparable to ingestion-time bookkeeping. USD only applies
        # to Voyage; BGE / Qwen3-MLX are in-process.
        #
        # `target_table_names` deliberately does NOT add
        # `_PER_CORPUS_STORE` for the --all case: --all routes corpus
        # updates through vector-copy from per-doc reembed (see
        # `_resolve_reembed_targets`), so the corpus rows incur no
        # additional encoder work and counting them here would inflate
        # the operator's cost estimate ~2x for narrative-heavy corpora
        # (review finding #2). `--corpus` alone still encodes the
        # corpus directly, in which case `include_corpus=True` and the
        # corpus table is the only target.
        import lancedb

        from auto_research.extract.chunking import count_tokens

        # The dry-run reads source rows from the ACTIVE materialization —
        # that's what reembed_doc/reembed_corpus would re-encode. Without
        # an active pointer the operation is impossible; surface a one-
        # line remediation rather than silently reporting 0 tokens.
        active = read_active_materialization(rag_root)
        if active is None:
            raise click.UsageError(
                f"no active materialization under {rag_root}; reembed is "
                "impossible without a source. Build the initial "
                "materialization (via embed) or migrate a legacy layout, "
                "then promote it before dry-running reembed."
            )

        db = lancedb.connect(rag_root)
        tables_seen = 0
        total_rows = 0
        total_tokens = 0
        target_bases = list(per_doc_targets)
        if include_corpus:
            target_bases.append(_PER_CORPUS_STORE)
        for base in target_bases:
            name = versioned_table_name(base, active.version)
            if name not in db.table_names():
                click.echo(f"warn: table {name!r} not found under {rag_root}", err=True)
                continue
            tbl = db.open_table(name)
            df = tbl.to_pandas()
            tables_seen += 1
            total_rows += len(df)
            # `count_tokens(None)` would raise; guard against the
            # nullable-text schema slot so dry-run can't crash on a
            # malformed row (defensive; the embed path never writes
            # NULL text but the schema permits it).
            total_tokens += sum(
                count_tokens(t) for t in df["text"].tolist() if t is not None
            )
        rate = _VOYAGE_USD_PER_MTOK.get(adapter.model)
        if backend_lit in ("bge", "qwen3-mlx"):
            cost_str = "$0.0000 (in-process)"
        elif rate is None:
            cost_str = "unknown (no rate in _VOYAGE_USD_PER_MTOK)"
        else:
            # 4 decimal places: sub-cent estimates are real signal at
            # the small-batch scale this command is most often used for
            # (one doc dry-run is fractions of a cent). `:.2f` rounded
            # those to $0.00 and falsely advertised free runs (review
            # finding #6).
            usd = (total_tokens / 1_000_000) * rate
            cost_str = f"~${usd:.4f} @ ${rate:.4f}/MTok"
        click.echo(
            f"reembed dry-run: backend={backend_lit} model={adapter.model} "
            f"version={version_token} dim={new_dim} tables={tables_seen} "
            f"rows={total_rows} tokens~{total_tokens} cost~{cost_str}"
        )
        return

    # Live path: per-doc tables one by one, then the corpus narrative.
    # Each call is atomic per-table (LanceDB `create_table mode="overwrite"`);
    # a crash mid-iteration leaves earlier tables consistent and later ones
    # untouched, and a re-run is the recovery primitive.
    per_doc_ok = 0
    per_doc_rows = 0
    per_doc_failed: list[str] = []
    for name in per_doc_targets:
        try:
            n = adapter.reembed_doc(name)
        except Exception as exc:
            click.echo(
                f"warn: reembed_doc({name!r}) failed: {exc.__class__.__name__}: {exc}",
                err=True,
            )
            per_doc_failed.append(name)
            continue
        per_doc_ok += 1
        per_doc_rows += n

    corpus_rows: int | None = None
    if include_corpus:
        try:
            corpus_rows = adapter.reembed_corpus()
        except Exception as exc:
            click.echo(
                f"warn: reembed_corpus() failed: {exc.__class__.__name__}: {exc}",
                err=True,
            )

    click.echo(
        f"reembed: backend={backend_lit} model={adapter.model} "
        f"version={version_token} "
        f"per_doc_ok={per_doc_ok} per_doc_rows={per_doc_rows} "
        f"per_doc_failed={len(per_doc_failed)} "
        f"corpus_rows={corpus_rows if corpus_rows is not None else 'skipped'}"
    )
    if per_doc_failed or (include_corpus and corpus_rows is None):
        raise SystemExit(1)


@extract.command(
    "list-materializations",
    help=(
        "List materialization versions present under --rag-root, with table "
        "counts and an active flag. Materialization versions are the "
        "8-hex-char hash of (chunker, contextual-prompt, embed-model) "
        "versions the rows were produced under."
    ),
)
@click.option(
    "--rag-root",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=_DEFAULT_RAG_ROOT,
    show_default=True,
    help="LanceDB root directory containing per-doc and corpus tables.",
)
def extract_list_materializations(rag_root: Path) -> None:
    try_init_telemetry()
    materializations = list_materializations(rag_root)
    if not materializations:
        click.echo(f"no materializations found under {rag_root}")
        return
    for m in materializations:
        active_marker = " (active)" if m.is_active else ""
        click.echo(
            f"{m.version}{active_marker} tables={m.table_count}"
        )


_NARRATIVE_DOC_TYPES: frozenset[str] = frozenset({"10-K", "10-Q", "transcript"})


def _expected_doc_ids_from_manifest(manifest_path: Path) -> set[str]:
    """All `status='ok'` doc_ids from the manifest, regardless of source.

    `manifest_mod.existing_doc_ids` requires an explicit `source` kwarg
    and is designed for fetch-loop dedup. Promotion validation needs the
    union across every source the project ingests — today just `edgar`,
    but `fmp` and future sources land in the same manifest under
    different `source` values and MUST be covered by the completeness
    check or partial namespaces could be promoted as the source set
    grows.
    """
    import pyarrow.parquet as pq

    if not manifest_path.exists():
        return set()
    table = pq.read_table(manifest_path, columns=["doc_id", "status"])
    return {
        d
        for d, st in zip(
            table.column("doc_id").to_pylist(),
            table.column("status").to_pylist(),
            strict=True,
        )
        if st == "ok"
    }


def _validate_promotion_candidate(
    rag_root: Path,
    *,
    version: str,
    manifest_path: Path,
) -> tuple[ActiveMaterialization, int]:
    """Validate that `version` is fully populated under `rag_root` and the
    vector dims are consistent. Returns the `ActiveMaterialization` record
    that should land in the pointer and the manifest doc count.

    Validation gates:

    1. **Namespace completeness.** Every `status='ok'` doc_id in the
       manifest (across all sources) must have a
       `{doc_id}__{version}.lance` table. Partial namespaces are refused
       with the list of missing doc_ids — promoting a partial namespace
       would silently regress corpus coverage at query time.
    2. **Corpus narrative presence.** If any of the per-doc tables has
       a narrative `doc_type` (10-K, 10-Q, transcript), the per-corpus
       narrative table at this version MUST exist; otherwise Signal
       A1's `store='corpus_narrative'` queries fail at runtime with
       `FileNotFoundError` against a freshly-promoted namespace.
    3. **Dim consistency.** Every table under this version must declare
       the same vector dim. Promoting a mixed-dim namespace would
       produce silent query failures at the boundary between docs.
    4. **embed_model_version row stamp consistency.** Every row in every
       table under this version must carry the same
       `embed_model_version` stamp; this is the value that lands in the
       active pointer for the read-path mismatch guard. Heterogeneity
       here is a bug in the build path; refusing here surfaces it
       early.
    5. **Non-empty namespace.** At least one row must exist somewhere
       in the namespace — otherwise the embed_model_version stamp is
       unrecoverable and the pointer would land with a placeholder no
       adapter could ever match.
    """
    import lancedb

    if not rag_root.exists():
        raise click.UsageError(f"--rag-root {rag_root} does not exist")
    if not manifest_path.exists():
        raise click.UsageError(f"--manifest-path {manifest_path} does not exist")

    db = lancedb.connect(rag_root)
    table_names = set(db.table_names())

    expected_doc_ids = _expected_doc_ids_from_manifest(manifest_path)
    missing: list[str] = []
    for doc_id in sorted(expected_doc_ids):
        if versioned_table_name(doc_id, version) not in table_names:
            missing.append(doc_id)
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise click.UsageError(
            f"materialization {version!r} is incomplete relative to "
            f"{manifest_path}: {len(missing)}/{len(expected_doc_ids)} "
            f"doc_id(s) lack a versioned table. Missing: [{preview}{more}]. "
            "Run the embed/reembed sweep to fill the gap before promoting."
        )

    # Dim + embed_model_version_stamp consistency across every table at
    # this version (including the optional corpus narrative table). The
    # corpus is appended LAST so any narrative-bearing per-doc table is
    # observed first and establishes the reference dim / embed_model
    # value before the corpus is compared against it.
    candidate_bases = sorted(expected_doc_ids)
    candidate_bases.append(_PER_CORPUS_STORE)
    dim_seen: int | None = None
    embed_model_version_seen: str | None = None
    narrative_doc_present = False
    for base in candidate_bases:
        name = versioned_table_name(base, version)
        if name not in table_names:
            continue  # corpus presence handled separately below
        tbl = db.open_table(name)
        field_type = tbl.schema.field("vector").type
        dim = int(field_type.list_size)
        if dim_seen is None:
            dim_seen = dim
        elif dim != dim_seen:
            raise click.UsageError(
                f"dim mismatch in materialization {version!r}: "
                f"{name!r} is {dim}-dim but earlier tables were "
                f"{dim_seen}-dim. Refusing to promote a mixed-vector-space "
                "namespace."
            )
        # Load just the two columns we validate against — full table
        # to_pandas would be wasteful at backfill scope. The stamp check
        # is intentionally row-level (Codex review, PR #73): head(1)
        # would let a single table with internally-mixed
        # `embed_model_version` stamps (a build-path bug or manual
        # LanceDB ops) pass validation, and the resulting active pointer
        # would advertise one vector space while queries silently read
        # from two.
        df = tbl.to_pandas()
        if len(df) == 0:
            continue
        emv_unique = set(df["embed_model_version"].unique())
        if len(emv_unique) > 1:
            raise click.UsageError(
                f"embed_model_version stamp NOT uniform within {name!r}: "
                f"{sorted(emv_unique)!r}. A single table must contain rows "
                "from exactly one vector space — refusing to promote a "
                "namespace where any table is internally mixed."
            )
        emv = next(iter(emv_unique))
        if embed_model_version_seen is None:
            embed_model_version_seen = emv
        elif emv != embed_model_version_seen:
            raise click.UsageError(
                f"embed_model_version stamp mismatch in materialization "
                f"{version!r}: {name!r} carries {emv!r} but earlier tables "
                f"carry {embed_model_version_seen!r}. Refusing to promote."
            )
        if (
            base != _PER_CORPUS_STORE
            and str(df["doc_type"].iloc[0]) in _NARRATIVE_DOC_TYPES
        ):
            narrative_doc_present = True

    if embed_model_version_seen is None:
        # No table had any rows. Refuse: the pointer's embed_model_version
        # stamp is unrecoverable, and an "unknown:unknown:unknown"
        # placeholder would make every subsequent query fail the
        # read-path mismatch guard. The operator likely embedded a
        # manifest whose docs all quarantined; surface that.
        raise click.UsageError(
            f"materialization {version!r} contains no rows in any table "
            f"under {rag_root}. Refusing to promote a fully-empty "
            "namespace — the active pointer's embed_model_version stamp "
            "is unrecoverable and the read-path mismatch guard would "
            "reject every subsequent query. Verify the embed sweep "
            "actually populated rows."
        )

    corpus_name = versioned_table_name(_PER_CORPUS_STORE, version)
    if narrative_doc_present and corpus_name not in table_names:
        raise click.UsageError(
            f"materialization {version!r} contains narrative documents "
            f"(10-K / 10-Q / transcript) but no {corpus_name!r} table. "
            "Signal A1's cross-doc retrieval queries the corpus store "
            "and would fail with FileNotFoundError against this "
            "namespace. Re-run the embed sweep to lazily-create the "
            "corpus, or delete the narrative per-doc tables if cross-doc "
            "retrieval is not required."
        )

    active = ActiveMaterialization(
        version=version,
        embed_model_version=embed_model_version_seen,
        promoted_at=now_utc_iso(),
        manifest_count=len(expected_doc_ids),
    )
    return active, len(expected_doc_ids)


@extract.command(
    "promote-materialization",
    help=(
        "Atomically promote a materialization version to be the active "
        "read namespace. Validates namespace completeness against the "
        "ingest manifest AND vector-dim consistency before flipping. The "
        "flip writes data/rag/active_materialization.json via tmp + "
        "rename so a crash mid-flip preserves the previous pointer."
    ),
)
@click.option("--version", "version", required=True, help="Materialization version slug to promote.")
@click.option(
    "--rag-root",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=_DEFAULT_RAG_ROOT,
    show_default=True,
    help="LanceDB root directory.",
)
@click.option(
    "--manifest-path",
    type=click.Path(dir_okay=False, exists=True, path_type=Path, resolve_path=True),
    default=_DEFAULT_MANIFEST,
    show_default=True,
    help="Ingest manifest Parquet file, source of truth for expected doc_id set.",
)
def extract_promote_materialization(
    version: str, rag_root: Path, manifest_path: Path
) -> None:
    try_init_telemetry()
    active, n_docs = _validate_promotion_candidate(
        rag_root, version=version, manifest_path=manifest_path
    )
    write_active_materialization(rag_root, active)
    append_promotion_history(rag_root, active)
    click.echo(
        f"promoted: version={active.version} "
        f"embed_model_version={active.embed_model_version} "
        f"manifest_count={n_docs}"
    )


@extract.command(
    "gc-materialization",
    help=(
        "Drop on-disk tables for old non-active materialization versions. "
        "Keeps the currently-active version unconditionally plus the "
        "(--keep-last N - 1) most recent previously-promoted versions "
        "from the promotion history. Default --keep-last 2 = active + 1 "
        "previous, enough for an instant rollback."
    ),
)
@click.option(
    "--keep-last",
    type=int,
    default=2,
    show_default=True,
    help="Number of versions to keep (current active counts as one).",
)
@click.option(
    "--rag-root",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=_DEFAULT_RAG_ROOT,
    show_default=True,
    help="LanceDB root directory.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report which tables would be removed without touching the disk.",
)
def extract_gc_materialization(
    keep_last: int, rag_root: Path, dry_run: bool
) -> None:
    try_init_telemetry()
    import shutil

    if keep_last < 1:
        raise click.UsageError("--keep-last must be >= 1")
    if not rag_root.exists():
        raise click.UsageError(f"--rag-root {rag_root} does not exist")

    active = read_active_materialization(rag_root)
    history = read_promotion_history(rag_root)
    # Refuse to GC when no active pointer exists: otherwise the keep
    # list could be empty (fresh install + built materialization but
    # not yet promoted) and the removal pass would wipe every table on
    # disk. The fix is operator-explicit: promote first, then GC.
    # Codex review (PR #73): "destructive default for a common setup".
    if active is None:
        raise click.UsageError(
            f"no active materialization under {rag_root}; refusing to "
            "GC without an anchor — a fresh-install or not-yet-promoted "
            "namespace would lose every table. Promote a version first, "
            "then re-run gc-materialization."
        )
    # Walk history newest-to-oldest, dedup repeated versions (an operator
    # may promote-then-demote-then-repromote), keep up to N distinct.
    keep: list[str] = [active.version]
    for record in reversed(history):
        if record.version in keep:
            continue
        if len(keep) >= keep_last:
            break
        keep.append(record.version)
    keep_set = set(keep)

    materializations = list_materializations(rag_root)
    to_remove: list[tuple[str, str]] = []  # (version, table_name)
    for m in materializations:
        if m.version in keep_set:
            continue
        for base in m.bases:
            to_remove.append((m.version, versioned_table_name(base, m.version)))

    if dry_run:
        click.echo(
            f"gc dry-run: keep={sorted(keep_set)} "
            f"would_remove_tables={len(to_remove)}"
        )
        for version, name in to_remove:
            click.echo(f"  would rm: {name} (version={version})")
        return

    removed = 0
    failed: list[tuple[str, str]] = []
    for _, name in to_remove:
        path = rag_root / f"{name}.lance"
        if not path.exists():
            continue
        # Wrap each rmtree so a single locked / read-only / EIO table
        # doesn't abort the whole sweep. The CLI reports both removed and
        # failed counts so operators can re-run safely after fixing the
        # underlying cause (closing a stale LanceDB process, fixing
        # permissions). Exit nonzero if any failed so make-target wiring
        # surfaces the partial-failure.
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            click.echo(
                f"warn: rmtree({path}) failed: "
                f"{exc.__class__.__name__}: {exc}",
                err=True,
            )
            failed.append((name, str(exc)))
    click.echo(
        f"gc-materialization: kept={sorted(keep_set)} "
        f"removed_tables={removed} failed_tables={len(failed)}"
    )
    if failed:
        raise SystemExit(1)


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


cli.add_command(eval_group)
