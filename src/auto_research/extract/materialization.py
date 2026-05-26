"""Materialization-versioned LanceDB layout helpers.

Eliminates the inconsistent-corpus window that a write-in-place re-embed
exhibits at backfill scope: at 1000-name x 10-year scope a single re-embed
pass against Voyage's constrained tier takes ~9 hours of wall-clock, during
which the per-doc and per-corpus tables would contain mixed-vector-space
rows (some new-model, some old-model) — the exact failure mode the
embedding-vector-space-consistency rule warns against. An atomic active-
pointer flip is the only correct way to migrate at this scale.

Two on-disk artifacts back the layout:

- LanceDB tables named `{base}__{materialization_version}` where `base` is
  the per-doc `doc_id` or the per-corpus `_corpus_narrative` sentinel.
  Multiple materializations can coexist; building a new one never touches
  the live one.
- `data/rag/active_materialization.json` — a small JSON pointer naming the
  materialization that queries should read from. Atomically updated via the
  shared `atomic_write_text` (tmp + fsync + rename + dir-fsync).
- `data/rag/promotion_history.json` — append-only list of past
  promotions. Read by `gc-materialization` for chronological "keep last N"
  semantics.

`materialization_version` is `sha256(chunker_version | contextual_prompt_
version | embed_model_version)` truncated to 8 hex characters: short
enough to embed in table names, long enough to collision-resist across
the dozen-or-so versions a project will ever produce. The pipe separator
makes the hash function injective over the three independent contract
strings (no ambiguity from "ab" + "cd" vs "abc" + "d").

`LEGACY_VERSION = "v0"` is reserved for tables produced by the pre-
versioning code path. The one-shot migration script renames legacy
tables into this namespace and seeds the initial pointer; this version
string is the sentinel the read path recognizes when no current-shape
materialization is yet promoted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from auto_research._io import atomic_write_text

ACTIVE_FILE_NAME: str = "active_materialization.json"
PROMOTION_HISTORY_FILE_NAME: str = "promotion_history.json"

# Sentinel for the pre-versioning materialization. The migration script
# renames every legacy `{base}.lance` table to `{base}__v0.lance`; queries
# resolved against this version go through the same code path as any
# other version.
LEGACY_VERSION: str = "v0"

# Separator between the table's base name and its materialization version.
# Double-underscore is unambiguous against any character we ever see in a
# doc_id: EDGAR accession numbers are `dddddddddd-dd-dddddd` (digits +
# single hyphens), tickers are uppercase letters, and the per-corpus
# sentinel `_corpus_narrative` contains only single underscores.
_VERSION_SEPARATOR: str = "__"


def compute_materialization_version(
    chunker_version: str,
    contextual_prompt_version: str,
    embed_model_version: str,
) -> str:
    """Return the 8-hex-char materialization version for a `(chunker,
    contextual-prompt, embed-model)` triple.

    Pure function: same inputs → same hash, no I/O. Pipe-separated so the
    hash is unambiguous over the three strings (otherwise "ab" + "cd" and
    "abc" + "d" would collide).
    """
    payload = f"{chunker_version}|{contextual_prompt_version}|{embed_model_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def versioned_table_name(base: str, version: str) -> str:
    """Join a table's base name to a materialization version slug.

    `base` is either a doc_id (per-doc store) or `_corpus_narrative` (per-
    corpus store). The same separator is used everywhere so
    `split_versioned_table_name` can invert this.
    """
    if not base:
        raise ValueError("base table name must be non-empty")
    if not version:
        raise ValueError("materialization version must be non-empty")
    if _VERSION_SEPARATOR in version:
        raise ValueError(
            f"materialization version must not contain {_VERSION_SEPARATOR!r}; "
            f"got {version!r}"
        )
    return f"{base}{_VERSION_SEPARATOR}{version}"


def split_versioned_table_name(table_name: str) -> tuple[str, str] | None:
    """Reverse of `versioned_table_name`. Returns None if `table_name` is
    not in the `{base}__{version}` shape — lets callers iterate every
    LanceDB table in a rag_root and skip the unrecognized ones gracefully
    (the pre-migration legacy layout, partial scratch directories, etc.).

    The split is anchored on the LAST `__` so doc_ids containing a single
    `_` (none today, but defensive) round-trip cleanly.
    """
    idx = table_name.rfind(_VERSION_SEPARATOR)
    if idx <= 0:
        return None
    base = table_name[:idx]
    version = table_name[idx + len(_VERSION_SEPARATOR) :]
    if not base or not version:
        return None
    return (base, version)


@dataclass(frozen=True)
class ActiveMaterialization:
    """The contents of `active_materialization.json`.

    `embed_model_version` is the full token from
    `embeddings.embed_model_version()` (e.g., `"voyage:voyage-finance-2:v1"`)
    so the read-path mismatch guard is unambiguous: an adapter with a
    different backend / model / version tag than the active pointer must
    raise loudly rather than silently degrade onto incompatible vectors.

    `manifest_count` is the number of `(source='edgar', status='ok')`
    doc_ids the promotion was validated against. Stored for forensic
    purposes — operators reviewing past promotions can correlate against
    the manifest snapshot at the time.
    """

    version: str
    embed_model_version: str
    promoted_at: str  # ISO-8601 UTC
    manifest_count: int


def _path_active(rag_root: Path) -> Path:
    return rag_root / ACTIVE_FILE_NAME


def _path_history(rag_root: Path) -> Path:
    return rag_root / PROMOTION_HISTORY_FILE_NAME


def read_active_materialization(rag_root: Path) -> ActiveMaterialization | None:
    """Return the current active pointer or None if no pointer is written.

    None is the "fresh install or pre-migration" state: callers fall back
    to whatever default they want (typically the adapter's own
    materialization version, so a single-shot build+query session works
    without an explicit promote step).

    A corrupted or partially-written JSON file is treated as a hard error
    (not None) — silently degrading to "no pointer" on parse failure would
    make a manual edit or filesystem corruption indistinguishable from
    "fresh install" and mask the very condition that needs operator
    intervention. The error names the file so the remediation is obvious.
    """
    p = _path_active(rag_root)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"active materialization pointer at {p} is not valid JSON "
            f"({exc.__class__.__name__}: {exc}). Inspect the file by hand; "
            "if corruption is recent, restore from the most recent "
            "promotion_history.json entry."
        ) from exc
    return ActiveMaterialization(
        version=raw["version"],
        embed_model_version=raw["embed_model_version"],
        promoted_at=raw["promoted_at"],
        manifest_count=int(raw["manifest_count"]),
    )


def write_active_materialization(
    rag_root: Path, active: ActiveMaterialization
) -> None:
    """Atomically write the active pointer.

    Discipline: `atomic_write_text` writes a hidden tmp sibling, fsyncs
    it, then `os.replace`s onto the destination — the rename is atomic
    at the POSIX layer. A crash between tmp-write and rename leaves the
    previous pointer intact (the property the issue's "failure-mid-flip"
    acceptance criterion is asserting).
    """
    rag_root.mkdir(parents=True, exist_ok=True)
    body = json.dumps(asdict(active), indent=2, sort_keys=True) + "\n"
    atomic_write_text(_path_active(rag_root), body)


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a `Z` suffix.

    Centralized so tests can monkeypatch a single symbol when they need to
    assert against a known promotion timestamp.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_promotion_history(
    rag_root: Path, active: ActiveMaterialization
) -> None:
    """Append `active` to the promotion-history file.

    The history is read by `gc-materialization` to order versions by
    promotion time (the materialization-version hash itself is random and
    can't be ordered). Same `atomic_write_text` durability discipline as
    the active pointer; we read-modify-write rather than streaming-append
    because the file is small (a few KB even after years of promotions)
    and the simpler primitive is easier to reason about.
    """
    history = read_promotion_history(rag_root)
    history.append(active)
    body = (
        json.dumps([asdict(a) for a in history], indent=2, sort_keys=True) + "\n"
    )
    atomic_write_text(_path_history(rag_root), body)


def read_promotion_history(rag_root: Path) -> list[ActiveMaterialization]:
    """Return the recorded promotion history, oldest first.

    Returns an empty list for a fresh rag_root. The list is in append
    order; the same version slug may appear more than once if an operator
    promoted-then-demoted-then-repromoted (no special handling, the most
    recent occurrence wins for "current" semantics).

    A corrupted history file raises (same policy as the active pointer):
    silently treating it as empty would let `gc-materialization` start
    deleting tables it should be keeping.
    """
    p = _path_history(rag_root)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"promotion history at {p} is not valid JSON "
            f"({exc.__class__.__name__}: {exc}). Inspect the file by "
            "hand; the active pointer is unaffected if it's intact."
        ) from exc
    return [
        ActiveMaterialization(
            version=row["version"],
            embed_model_version=row["embed_model_version"],
            promoted_at=row["promoted_at"],
            manifest_count=int(row["manifest_count"]),
        )
        for row in raw
    ]


@dataclass(frozen=True)
class MaterializationInfo:
    """One materialization's on-disk footprint, returned by `list_materializations`.

    `table_count` is the number of `{base}__{version}.lance` directories
    discovered at this version (per-doc + per-corpus combined; the caller
    knows the split via the bases). `is_active` is true iff the active
    pointer names this version.
    """

    version: str
    table_count: int
    bases: tuple[str, ...]
    is_active: bool


def list_materializations(rag_root: Path) -> list[MaterializationInfo]:
    """Enumerate every materialization version present under `rag_root`.

    Scans `*.lance` directories, groups by the version suffix, returns
    one record per version with the list of bases (doc_ids and/or
    `_corpus_narrative`) at that version. Sorted by version slug for
    deterministic output.
    """
    if not rag_root.exists():
        return []
    active = read_active_materialization(rag_root)
    active_version = active.version if active is not None else None
    by_version: dict[str, list[str]] = {}
    for entry in rag_root.iterdir():
        if not (entry.is_dir() and entry.suffix == ".lance"):
            continue
        split = split_versioned_table_name(entry.stem)
        if split is None:
            continue
        base, version = split
        by_version.setdefault(version, []).append(base)
    return [
        MaterializationInfo(
            version=version,
            table_count=len(bases),
            bases=tuple(sorted(bases)),
            is_active=(version == active_version),
        )
        for version, bases in sorted(by_version.items())
    ]


__all__ = [
    "ACTIVE_FILE_NAME",
    "LEGACY_VERSION",
    "PROMOTION_HISTORY_FILE_NAME",
    "ActiveMaterialization",
    "MaterializationInfo",
    "append_promotion_history",
    "compute_materialization_version",
    "list_materializations",
    "now_utc_iso",
    "read_active_materialization",
    "read_promotion_history",
    "split_versioned_table_name",
    "versioned_table_name",
    "write_active_materialization",
]
