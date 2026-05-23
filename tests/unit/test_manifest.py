"""Unit tests for the append-only manifest ledger (Issue #5)."""

from __future__ import annotations

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


def test_read_missing_manifest_returns_empty_table(tmp_path: Path) -> None:
    table = manifest.read(tmp_path / "absent.parquet")
    assert table.num_rows == 0
    # Schema is still well-defined so downstream filters don't blow up.
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
    table = manifest.read(path)
    assert sorted(table.column("doc_id").to_pylist()) == ["A", "B"]


def test_contains_finds_existing_row(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    manifest.append(path, [_row(source="edgar", doc_id="X")])
    assert manifest.contains(path, source="edgar", doc_id="X") is True
    assert manifest.contains(path, source="edgar", doc_id="Y") is False
    # Source is part of the key — same doc_id under different source is distinct.
    assert manifest.contains(path, source="fmp", doc_id="X") is False


def test_contains_on_missing_manifest_returns_false(tmp_path: Path) -> None:
    assert manifest.contains(tmp_path / "nope.parquet", source="edgar", doc_id="X") is False
