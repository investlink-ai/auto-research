"""Unit tests for the append-only manifest ledger (Issue #5)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from auto_research.ingest import manifest


def _row(
    *,
    source: str = "edgar",
    entity_id: str = "0001045810",
    doc_id: str = "0001045810-24-000316",
    form_type: str = "10-K",
    sha: str = "a" * 64,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "source": source,
        "entity_id": entity_id,
        "doc_id": doc_id,
        "form_type": form_type,
        "event_datetime": datetime(2024, 2, 21, 16, 31, tzinfo=UTC),
        "fetched_at": datetime(2026, 5, 23, 0, 0, tzinfo=UTC),
        "content_sha256": sha,
        "path": f"data/raw/{source}/{entity_id}/2024/{doc_id}.htm",
        "status": status,
    }


# ---------- read / append round-trip ----------


def test_read_missing_manifest_returns_empty_table(tmp_path: Path) -> None:
    table = manifest.read(tmp_path / "absent.parquet")
    assert table.num_rows == 0
    assert "doc_id" in table.schema.names


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(doc_id="A"), _row(doc_id="B")])
    table = manifest.read(path)
    assert table.num_rows == 2
    assert sorted(table.column("doc_id").to_pylist()) == ["A", "B"]


def test_append_is_additive_not_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(doc_id="A")])
    manifest.append(path, [_row(doc_id="B")])
    assert sorted(manifest.read(path).column("doc_id").to_pylist()) == ["A", "B"]


def test_append_empty_list_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    manifest.append(path, [])
    assert not path.exists()


# ---------- contains() with status filter ----------


def test_contains_finds_ok_row_by_default(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(source="edgar", doc_id="X", status="ok")])
    assert manifest.contains(path, source="edgar", doc_id="X") is True
    assert manifest.contains(path, source="edgar", doc_id="Y") is False
    assert manifest.contains(path, source="fmp", doc_id="X") is False


def test_contains_excludes_error_rows_by_default(tmp_path: Path) -> None:
    """Default filter is status='ok' — error rows must NOT count as cached.

    Otherwise a single failure would become a permanent cache hit and
    block retries. INV-2 routes failures to data/quarantine/; the
    manifest must allow retrying them.
    """
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(source="edgar", doc_id="X", status="error")])
    assert manifest.contains(path, source="edgar", doc_id="X") is False


def test_contains_status_none_matches_any(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(source="edgar", doc_id="X", status="error")])
    assert manifest.contains(path, source="edgar", doc_id="X", status=None) is True


def test_contains_on_missing_manifest_returns_false(tmp_path: Path) -> None:
    assert manifest.contains(tmp_path / "nope.parquet", source="edgar", doc_id="X") is False


# ---------- existing_doc_ids snapshot ----------


def test_existing_doc_ids_returns_set_of_oks_only(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    manifest.append(
        path,
        [
            _row(doc_id="A", status="ok"),
            _row(doc_id="B", status="ok"),
            _row(doc_id="C", status="error"),
        ],
    )
    assert manifest.existing_doc_ids(path, source="edgar") == {"A", "B"}


def test_existing_doc_ids_status_none_returns_all(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(doc_id="A", status="error")])
    assert manifest.existing_doc_ids(path, source="edgar", status=None) == {"A"}


def test_existing_doc_ids_on_missing_manifest_returns_empty(tmp_path: Path) -> None:
    assert manifest.existing_doc_ids(tmp_path / "nope.parquet", source="edgar") == set()


# ---------- nullable event_datetime (forward-compat for FMP no-coverage rows) ----------


def test_append_accepts_null_event_datetime(tmp_path: Path) -> None:
    """Issue #6 (FMP) will write rows with event_datetime=None for gap-cases."""
    path = tmp_path / "m.parquet"
    row = _row(doc_id="X")
    row["event_datetime"] = None
    manifest.append(path, [row])
    table = manifest.read(path)
    assert table.column("event_datetime")[0].as_py() is None


# ---------- concurrent-write race (file lock) ----------


def test_concurrent_appends_dont_lose_rows(tmp_path: Path) -> None:
    """Two threads appending disjoint rows must both land in the manifest.

    Without the file lock, the read-modify-write pattern would drop
    one writer's batch silently. With `fcntl.flock(LOCK_EX)`, the
    writes serialise and all rows land.
    """
    path = tmp_path / "m.parquet"
    n_per_thread = 25
    n_threads = 4

    def writer(thread_id: int) -> None:
        rows = [_row(doc_id=f"T{thread_id}-{i}") for i in range(n_per_thread)]
        manifest.append(path, rows)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    table = manifest.read(path)
    assert table.num_rows == n_per_thread * n_threads
    doc_ids = set(table.column("doc_id").to_pylist())
    expected = {f"T{tid}-{i}" for tid in range(n_threads) for i in range(n_per_thread)}
    assert doc_ids == expected


# ---------- dedup-on-append (cross-process race guard) ----------


def test_append_dedups_against_existing_on_unique_keys(tmp_path: Path) -> None:
    """Cross-process snapshot race guard: the same (source, doc_id) can't double-insert.

    Two concurrent fetchers both snapshot an empty existing-set and
    both submit a row for accession 'X'. The first append writes it;
    the second sees it on disk under the file lock and drops the
    duplicate before writing.
    """
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(doc_id="X")])
    # Second append with the SAME doc_id should be a no-op.
    manifest.append(path, [_row(doc_id="X", sha="b" * 64)])
    table = manifest.read(path)
    assert table.num_rows == 1
    # First write wins — second one was dropped before the concat.
    assert table.column("content_sha256").to_pylist() == ["a" * 64]


def test_append_dedups_within_a_batch_against_existing(tmp_path: Path) -> None:
    """A batch where some rows are new and some are dups records only the new ones."""
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(doc_id="X")])
    manifest.append(path, [_row(doc_id="X"), _row(doc_id="Y"), _row(doc_id="Z")])
    assert sorted(manifest.read(path).column("doc_id").to_pylist()) == ["X", "Y", "Z"]


def test_append_with_unique_keys_none_allows_duplicates(tmp_path: Path) -> None:
    """Opting out of dedup is supported for true append-only audit ledgers."""
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(doc_id="X")])
    manifest.append(path, [_row(doc_id="X")], unique_keys=None)
    assert manifest.read(path).num_rows == 2
