"""Append-only Parquet ledger of fetched documents.

Per spec §6.1 the manifest is `(source, entity_id, doc_id, fetched_at,
content_sha256, status)` plus the source-specific PIT stamp
(`event_datetime`), form type, and on-disk path. It's the idempotency
boundary for every ingest module — calling `contains(...)` before a fetch
is what makes re-runs no-ops.

Writes are full-rewrite-then-rename: we read the existing table, append
new rows, write to a sibling tempfile, and `replace()`. At <1M rows
this is cheap and gives us atomicity for free. If/when the manifest
grows past that, swap to a partitioned dataset.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
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
        ("event_datetime", pa.timestamp("us", tz="UTC")),
        ("fetched_at", pa.timestamp("us", tz="UTC")),
        ("content_sha256", pa.string()),
        ("path", pa.string()),
        ("status", pa.string()),
    ]
)


def _empty_table() -> pa.Table:
    return pa.table({name: pa.array([], type=ty) for name, ty in zip(SCHEMA.names, SCHEMA.types, strict=True)})


def read(path: Path) -> pa.Table:
    """Read the manifest, or return an empty table with the canonical schema."""
    if not path.exists():
        return _empty_table()
    return pq.read_table(path)


def append(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append rows to the manifest. Atomic via write-then-rename."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_table = pa.Table.from_pylist(list(rows), schema=SCHEMA)
    combined = pa.concat_tables([read(path), new_table]) if path.exists() else new_table
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(combined, tmp)
    os.replace(tmp, path)


def contains(path: Path, *, source: str, doc_id: str) -> bool:
    """True iff a row exists with `(source, doc_id)` already recorded."""
    if not path.exists():
        return False
    table = pq.read_table(path, columns=["source", "doc_id"])
    sources = table.column("source").to_pylist()
    doc_ids = table.column("doc_id").to_pylist()
    return any(s == source and d == doc_id for s, d in zip(sources, doc_ids, strict=True))
