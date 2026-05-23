"""Append-only Parquet ledger of fetched documents.

The manifest is the idempotency boundary for every ingest module: calling
`existing_doc_ids(...)` (or `contains(...)`) before a fetch is what makes
re-runs no-ops. Schema is the single source of truth — see `SCHEMA`
below; any change to it requires an explicit migration step (no in-code
schema-tolerance layer is provided).

Durability: writes are full-rewrite-then-rename, guarded by an
`fcntl.flock` advisory lock and `os.fsync` on both the temp file and
the parent directory before the rename. The PID is embedded in the
temp filename so two writers can't trample each other's bytes even if
the lock is bypassed.

Cross-process idempotency: `append()` deduplicates new rows against
the on-disk manifest *inside the lock*, keyed by `unique_keys`
(default `(source, doc_id)`). This closes the snapshot-then-write
race where two concurrent fetchers both see an empty
`existing_doc_ids` and both submit the same row.

Concurrency: `flock(LOCK_EX)` serialises writers across processes on
the same machine. NFS / network filesystems vary in flock semantics —
single-machine fan-out is the supported workload.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading
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
# Dedup key includes `status` so future writers (FMP no_coverage rows;
# extract quarantine workflows per INV-2) can record an error row and
# then, on a successful retry, append a new ok row WITHOUT the retry's ok
# row being silently dropped by lock-time dedup against the prior error
# row. The EDGAR client today only writes status='ok'; this is the
# forward-looking shape for the source-shared manifest.
#
# Status DEMOTION (ok → quarantined) is not supported by simple `append`:
# both rows coexist and `existing_doc_ids(status=("ok",))` keeps reporting
# the doc as cached against the original ok row. Quarantine workflows
# that need to invalidate a previously-ok row must do an explicit
# manifest rewrite, not just an append.
DEFAULT_UNIQUE_KEYS: tuple[str, ...] = ("source", "doc_id", "status")


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


def append(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    unique_keys: Sequence[str] | None = DEFAULT_UNIQUE_KEYS,
) -> None:
    """Append rows to the manifest. Atomic + durable + process-safe.

    Inside the file lock, drops any incoming row whose `unique_keys`
    tuple already exists on disk. Default `unique_keys=("source",
    "doc_id", "status")` means same-status duplicates are deduped
    while a transition (e.g., `error → ok` after a successful retry)
    appends a new row instead of being silently swallowed. Pass
    `unique_keys=None` to skip the dedup.

    Raises `ValueError` if `unique_keys` references a column not in
    `SCHEMA` (otherwise a typo would silently corrupt the first
    write and crash on subsequent reads).

    Tmp file is removed on any failure between `tmp.write_bytes` and
    `os.replace` — so a partial fsync/disk-full doesn't leak hidden
    dotfiles into the parent directory.
    """
    if not rows:
        return
    if unique_keys is not None:
        # Reject empty sequence explicitly — it silently disables dedup
        # (same observable effect as `None`) and is almost always an
        # unintentional config bug (e.g., `tuple(filter(...))` returning empty).
        if not unique_keys:
            raise ValueError(
                "unique_keys must be a non-empty sequence or None; "
                "got empty sequence which would silently disable dedup"
            )
        unknown = [k for k in unique_keys if k not in SCHEMA.names]
        if unknown:
            raise ValueError(
                f"unique_keys contains columns not in SCHEMA: {unknown}; "
                f"valid columns are {SCHEMA.names}"
            )
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
            pa.concat_tables([existing, new_table]) if existing.num_rows > 0 else new_table
        )
        tmp = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            pq.write_table(combined, tmp)
            _fsync_path(tmp)
            os.replace(tmp, path)
        except BaseException:
            # Cleanup must NOT mask the original exception. unlink can itself
            # raise (PermissionError on read-only fs, EIO on a dying disk,
            # EBUSY) — swallow OSError from the cleanup and re-raise the
            # original write/fsync/replace failure with full fidelity.
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise
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
    table = pq.read_table(path, columns=["source", "doc_id", "status"])
    allowed = set(status) if status is not None else None
    return any(
        s == source and d == doc_id and (allowed is None or st in allowed)
        for s, d, st in zip(
            table.column("source").to_pylist(),
            table.column("doc_id").to_pylist(),
            table.column("status").to_pylist(),
            strict=True,
        )
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
    allowed = set(status) if status is not None else None
    return {
        d
        for s, d, st in zip(
            table.column("source").to_pylist(),
            table.column("doc_id").to_pylist(),
            table.column("status").to_pylist(),
            strict=True,
        )
        if s == source and (allowed is None or st in allowed)
    }
