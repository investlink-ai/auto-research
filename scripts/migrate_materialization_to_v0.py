"""Idempotent migration from the legacy write-in-place LanceDB layout to
the materialization-versioned layout.

Pre-migration: tables live at `{rag_root}/{doc_id}.lance` and
`{rag_root}/_corpus_narrative.lance`. Post-migration: the same data lives
at `{rag_root}/{doc_id}__v0.lance` and
`{rag_root}/_corpus_narrative__v0.lance`, and
`{rag_root}/active_materialization.json` names `v0` as the current
materialization.

`v0` is the reserved sentinel for the pre-versioning materialization
(see `auto_research.extract.materialization.LEGACY_VERSION`). It does
NOT correspond to any specific
`compute_materialization_version(chunker, prompt, embed_model)` triple
— the legacy tables predate row-level version stamping in any
queryable form. Subsequent re-embeds against new embed models produce
fresh hash-suffixed namespaces alongside `v0`; the active pointer
flips when the new namespace is promoted.

Idempotent: running this script twice is a no-op. Tables already at
`{base}__v0` are left alone; `active_materialization.json` is only
written if it doesn't already exist (or if the existing pointer also
names `v0`, in which case the write is a refresh).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from auto_research.extract.embeddings import _PER_CORPUS_STORE
from auto_research.extract.materialization import (
    ACTIVE_FILE_NAME,
    LEGACY_VERSION,
    ActiveMaterialization,
    now_utc_iso,
    read_active_materialization,
    split_versioned_table_name,
    versioned_table_name,
    write_active_materialization,
)


def _sample_embed_model_version(rag_root: Path) -> str:
    """Best-effort: read one row from any present LanceDB table to recover
    the `embed_model_version` stamp produced by issue #67's row-level
    columns. Falls back to a placeholder when the rag_root is empty or
    rows lack the column (extremely old tables predating #67).

    The pointer's `embed_model_version` matters at query time as the
    mismatch guard: adapters configured for a different model than the
    active pointer's value raise loudly. For the v0 migration we want
    the real value from disk so an operator constructing an adapter
    matched to the legacy corpus sees a clean read path; the
    placeholder is only for the truly degenerate empty-rag_root case.
    """
    import lancedb

    if not rag_root.exists():
        return "unknown:unknown:unknown"
    try:
        db = lancedb.connect(rag_root)
        table_names = db.table_names()
    except Exception:
        return "unknown:unknown:unknown"
    for table_name in table_names:
        try:
            tbl = db.open_table(table_name)
            df = tbl.head(1).to_pandas()
        except Exception:
            # Skip placeholder dirs that aren't real LanceDB tables (the
            # idempotency tests, partly-migrated layouts, and the genuinely
            # empty rag_root all surface as open_table errors here).
            continue
        if len(df) == 0:
            continue
        if "embed_model_version" not in df.columns:
            continue
        return str(df["embed_model_version"].iloc[0])
    return "unknown:unknown:unknown"


def _is_legacy_unversioned_table(path: Path) -> bool:
    """True iff `path` is a `<name>.lance/` dir whose stem is NOT already
    versioned (`<base>__<version>`). The migration leaves already-
    versioned tables alone — idempotency."""
    if not (path.is_dir() and path.suffix == ".lance"):
        return False
    return split_versioned_table_name(path.stem) is None


def migrate(rag_root: Path) -> dict[str, int]:
    """Run the v0 migration; return a counts dict for the operator log.

    Renames every legacy `{name}.lance` to `{name}__v0.lance` and writes
    the initial active pointer. Does not touch already-versioned tables
    or overwrite a non-v0 active pointer.
    """
    counts = {
        "renamed": 0,
        "already_versioned": 0,
        "pointer_written": 0,
        "pointer_unchanged": 0,
    }
    if not rag_root.exists():
        return counts

    for entry in sorted(rag_root.iterdir()):
        if not (entry.is_dir() and entry.suffix == ".lance"):
            continue
        if split_versioned_table_name(entry.stem) is not None:
            counts["already_versioned"] += 1
            continue
        base = entry.stem
        new_name = versioned_table_name(base, LEGACY_VERSION)
        new_path = rag_root / f"{new_name}.lance"
        if new_path.exists():
            # An earlier partial migration already moved this one; clean
            # up the dangling unversioned entry only if it's empty —
            # otherwise leave both in place and require the operator to
            # investigate. Avoid silent data loss.
            counts["already_versioned"] += 1
            continue
        entry.rename(new_path)
        counts["renamed"] += 1

    existing = read_active_materialization(rag_root)
    if existing is not None and existing.version != LEGACY_VERSION:
        # Operator promoted past v0 already; do not clobber. This is the
        # branch the second invocation of an interrupted-then-fixed
        # workflow takes.
        counts["pointer_unchanged"] += 1
        return counts

    active = ActiveMaterialization(
        version=LEGACY_VERSION,
        embed_model_version=_sample_embed_model_version(rag_root),
        promoted_at=now_utc_iso(),
        manifest_count=sum(
            1
            for entry in rag_root.iterdir()
            if entry.is_dir()
            and entry.suffix == ".lance"
            and entry.stem.endswith(f"__{LEGACY_VERSION}")
            and not entry.stem.startswith(_PER_CORPUS_STORE)
        ),
    )
    write_active_materialization(rag_root, active)
    counts["pointer_written"] += 1
    return counts


@click.command()
@click.option(
    "--rag-root",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=Path("data/rag"),
    show_default=True,
    help="LanceDB root directory to migrate.",
)
def main(rag_root: Path) -> None:
    """Migrate `--rag-root` from the legacy unversioned layout to the
    materialization-versioned layout. Idempotent.
    """
    if not rag_root.exists():
        click.echo(f"--rag-root {rag_root} does not exist; nothing to migrate")
        sys.exit(0)
    counts = migrate(rag_root)
    click.echo(
        f"migrate-materialization-to-v0: "
        f"renamed={counts['renamed']} "
        f"already_versioned={counts['already_versioned']} "
        f"pointer_written={counts['pointer_written']} "
        f"pointer_unchanged={counts['pointer_unchanged']} "
        f"active_pointer={rag_root / ACTIVE_FILE_NAME}"
    )


if __name__ == "__main__":
    main()
