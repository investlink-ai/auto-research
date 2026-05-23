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

Concurrency: `flock(LOCK_EX)` serialises writers across processes on
the same machine. NFS / network filesystems vary in flock semantics —
single-machine fan-out is the supported workload. Document accordingly.

Schema evolution: `pa.concat_tables(..., promote_options="default")`
accepts new nullable columns added in future versions, so a stale
on-disk manifest doesn't brick the upgrade. Reordering / renaming
columns is still a manual migration.
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


def _empty_table() -> pa.Table:
    return pa.table(
        {name: pa.array([], type=ty) for name, ty in zip(SCHEMA.names, SCHEMA.types, strict=True)}
    )


def read(path: Path) -> pa.Table:
    """Read the manifest, or return an empty table with the canonical schema."""
    if not path.exists():
        return _empty_table()
    return pq.read_table(path)


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


def append(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append rows to the manifest. Atomic + durable + process-safe."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_table = pa.Table.from_pylist(list(rows), schema=SCHEMA)
    with _exclusive_lock(path):
        existing = read(path)
        # promote_options="default" lets a manifest written under an older
        # schema (fewer columns, all nullable) survive a future column
        # addition. Renames and reorderings still require migration.
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


def _matches_status(value: str, allowed: Iterable[str] | None) -> bool:
    if allowed is None:
        return True
    return value in set(allowed)


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
    """
    if not path.exists():
        return False
    table = pq.read_table(path, columns=["source", "doc_id", "status"])
    sources = table.column("source").to_pylist()
    doc_ids = table.column("doc_id").to_pylist()
    statuses = table.column("status").to_pylist()
    return any(
        s == source and d == doc_id and _matches_status(st, status)
        for s, d, st in zip(sources, doc_ids, statuses, strict=True)
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
    """
    if not path.exists():
        return set()
    table = pq.read_table(path, columns=["source", "doc_id", "status"])
    sources = table.column("source").to_pylist()
    doc_ids = table.column("doc_id").to_pylist()
    statuses = table.column("status").to_pylist()
    allowed = set(status) if status is not None else None
    return {
        d
        for s, d, st in zip(sources, doc_ids, statuses, strict=True)
        if s == source and (allowed is None or st in allowed)
    }
