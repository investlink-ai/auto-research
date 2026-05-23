"""Append-only Parquet ledger of fetched documents.

Per spec §6.1 the manifest is `(source, entity_id, doc_id, fetched_at,
content_sha256, status)` plus the source-specific PIT stamp
(`event_datetime`), form type, and on-disk path. It's the idempotency
boundary for every ingest module — calling `contains(...)` /
`existing_doc_ids(...)` before a fetch is what makes re-runs no-ops.

Durability: writes are full-rewrite-then-rename, guarded by an
`fcntl.flock` advisory lock and `os.fsync` on both the temp file and
the parent directory before the rename. The PID is embedded in the
temp filename so two writers can't trample each other's bytes even if
the lock is bypassed (e.g., on a filesystem that doesn't honor flock).

Cross-process idempotency: `append()` deduplicates new rows against
the on-disk manifest *inside the lock*, keyed by `unique_keys`
(default `(source, doc_id)`). This closes the read-then-write race
where two concurrent fetchers each snapshot an empty `existing_doc_ids`,
both fetch the same documents, and both attempt to append — without
dedup-on-append, the manifest would silently grow duplicate rows.

Schema evolution: `pa.concat_tables(..., promote_options="default")`
accepts new nullable columns added in future versions. The READ path
(`contains`, `existing_doc_ids`) is also defensive — projects only
the columns that actually exist on disk, so a manifest written under
an older schema (e.g., before `status` was added) doesn't raise
during a snapshot scan. Adding *non-nullable* columns or renaming
columns still requires manual migration.

Concurrency: `flock(LOCK_EX)` serialises writers across processes on
the same machine. NFS / network filesystems vary in flock semantics —
single-machine fan-out is the supported workload.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA: pa.Schema = pa.schema(
    [
        ("source", pa.string()),
        ("entity_id", pa.string()),
        ("doc_id", pa.string()),
        ("form_type", pa.string()),
        # event_datetime is nullable: FMP "no_coverage" rows have no PIT stamp.
        ("event_datetime", pa.timestamp("us", tz="UTC")),
        ("fetched_at", pa.timestamp("us", tz="UTC")),
        ("content_sha256", pa.string()),
        ("path", pa.string()),
        ("status", pa.string()),
    ]
)

DEFAULT_OK_STATUSES: tuple[str, ...] = ("ok",)
DEFAULT_UNIQUE_KEYS: tuple[str, ...] = ("source", "doc_id")


def _empty_table() -> pa.Table:
    return pa.table(
        {name: pa.array([], type=ty) for name, ty in zip(SCHEMA.names, SCHEMA.types, strict=True)}
    )


def read(path: Path) -> pa.Table:
    """Read the manifest, or return an empty table with the canonical schema."""
    if not path.exists():
        return _empty_table()
    return pq.read_table(path)


def _read_columns_lenient(path: Path, want: Sequence[str]) -> dict[str, list[Any]]:
    """Project `want` columns from the on-disk manifest, defaulting missing ones to None.

    Defensive against schema evolution: a manifest written under an
    older SCHEMA (e.g., before `status` existed) won't raise; missing
    columns yield an all-None list of the same length as the table.
    """
    if not path.exists():
        return {col: [] for col in want}
    schema_on_disk = pq.read_schema(path)
    available = [col for col in want if col in schema_on_disk.names]
    table = pq.read_table(path, columns=available) if available else pq.read_table(path)
    n = table.num_rows
    out: dict[str, list[Any]] = {}
    for col in want:
        if col in schema_on_disk.names:
            out[col] = table.column(col).to_pylist()
        else:
            out[col] = [None] * n
    return out


@contextmanager
def _exclusive_lock(path: Path) -> Any:
    """Process-level exclusive lock on a sibling `.lock` file.

    Held for the entire read-modify-write of `append()`. The lock file
    is created on demand and never removed (removing it races with
    other waiters trying to open it).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with open(lock_path, "w") as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)


def _fsync_path(path: Path) -> None:
    """fsync the file and its parent dir so a power loss after `os.replace` doesn't leave a 0-byte parquet."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def append(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    unique_keys: Sequence[str] | None = DEFAULT_UNIQUE_KEYS,
) -> None:
    """Append rows to the manifest. Atomic + durable + process-safe.

    Inside the file lock, drops any incoming row whose `unique_keys`
    tuple already exists on disk — so two concurrent processes that
    both snapshot an empty existing-set and both try to append the
    same `(source, doc_id)` can't produce duplicate ledger rows. Pass
    `unique_keys=None` to skip the dedup (rarely useful — only for
    append-only audit logs where every row really is unique).
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(path):
        existing = read(path)
        rows_to_write = list(rows)
        if unique_keys and existing.num_rows > 0:
            cols = [existing.column(k).to_pylist() for k in unique_keys]
            existing_keys = set(zip(*cols, strict=True))
            rows_to_write = [
                row
                for row in rows_to_write
                if tuple(row.get(k) for k in unique_keys) not in existing_keys
            ]
        if not rows_to_write:
            return
        new_table = pa.Table.from_pylist(rows_to_write, schema=SCHEMA)
        combined = (
            pa.concat_tables([existing, new_table], promote_options="default")
            if existing.num_rows > 0
            else new_table
        )
        tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
        pq.write_table(combined, tmp)
        _fsync_path(tmp)
        os.replace(tmp, path)
        # Re-fsync the parent dir to persist the rename itself.
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def contains(
    path: Path,
    *,
    source: str,
    doc_id: str,
    status: Iterable[str] | None = DEFAULT_OK_STATUSES,
) -> bool:
    """True iff a row exists with `(source, doc_id)` and an allowed status.

    `status=None` matches any status (useful for "is this doc_id ever
    recorded?"); the default `("ok",)` means error / quarantined rows
    do NOT count as cached, so a downstream retry path can re-fetch
    them without manual manifest surgery.

    Performance note: scans the manifest linearly per call. For loops
    over many doc_ids, snapshot once via `existing_doc_ids()` and do
    O(1) set membership in the caller's loop.
    """
    if not path.exists():
        return False
    cols = _read_columns_lenient(path, ("source", "doc_id", "status"))
    allowed = set(status) if status is not None else None
    return any(
        s == source
        and d == doc_id
        and (allowed is None or st is None or st in allowed)
        for s, d, st in zip(cols["source"], cols["doc_id"], cols["status"], strict=True)
    )


def existing_doc_ids(
    path: Path,
    *,
    source: str,
    status: Iterable[str] | None = DEFAULT_OK_STATUSES,
) -> set[str]:
    """Snapshot of doc_ids already recorded under `source` with allowed status.

    Use this once at the start of a fetch loop to avoid the O(N·M) cost
    of per-filing `contains()` calls. Returns an empty set if the
    manifest doesn't exist.

    Lenient against legacy schemas: a manifest written before the
    `status` column existed reads with status=None for every row,
    which (with the default `status=("ok",)` filter) treats every
    legacy row as if status='ok'. That matches the pre-status-filter
    semantics, preventing a schema upgrade from invalidating prior
    work.
    """
    if not path.exists():
        return set()
    cols = _read_columns_lenient(path, ("source", "doc_id", "status"))
    allowed = set(status) if status is not None else None
    # Treat legacy rows (status column missing → None) as 'ok' so a schema
    # upgrade doesn't silently invalidate every existing ledger entry.
    return {
        d
        for s, d, st in zip(cols["source"], cols["doc_id"], cols["status"], strict=True)
        if s == source and (allowed is None or st is None or st in allowed)
    }
