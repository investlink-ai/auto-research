"""Unit tests for `scripts/migrate_materialization_to_v0.py`.

The migration is a one-shot operator action and the script lives outside
`src/`; we import it via the `scripts/` directory path so the same logic
that the operator runs is the logic these tests exercise.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from auto_research.extract.materialization import (
    ACTIVE_FILE_NAME,
    LEGACY_VERSION,
    ActiveMaterialization,
    read_active_materialization,
    write_active_materialization,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "migrate_materialization_to_v0.py"
)


def _import_script() -> object:
    """Import the migration script as a module so its `migrate()` is
    callable from unit tests without spawning a subprocess.

    Uses an importlib spec rather than a top-level `from scripts.X import …`
    because `scripts/` is not a package (no __init__.py) and shouldn't
    become one just to enable testing — the script is for operators, the
    helpers under test are already covered in `materialization.py`.
    """
    spec = importlib.util.spec_from_file_location(
        "migrate_materialization_to_v0", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["migrate_materialization_to_v0"] = module
    spec.loader.exec_module(module)
    return module


def _touch_legacy_table(rag_root: Path, name: str) -> Path:
    """Create an empty `.lance/` directory with no version suffix — the
    legacy layout shape the migration is designed to rewrite.
    """
    path = rag_root / f"{name}.lance"
    path.mkdir(parents=True)
    return path


def test_migrate_renames_legacy_tables_to_v0_namespace(tmp_path: Path) -> None:
    script = _import_script()
    _touch_legacy_table(tmp_path, "doc-A")
    _touch_legacy_table(tmp_path, "doc-B")
    _touch_legacy_table(tmp_path, "_corpus_narrative")

    counts = script.migrate(tmp_path)  # type: ignore[attr-defined]

    assert counts["renamed"] == 3
    assert (tmp_path / "doc-A__v0.lance").exists()
    assert (tmp_path / "doc-B__v0.lance").exists()
    assert (tmp_path / "_corpus_narrative__v0.lance").exists()
    # Old unversioned dirs are gone.
    assert not (tmp_path / "doc-A.lance").exists()
    assert not (tmp_path / "_corpus_narrative.lance").exists()


def test_migrate_writes_active_pointer(tmp_path: Path) -> None:
    script = _import_script()
    _touch_legacy_table(tmp_path, "doc-A")
    script.migrate(tmp_path)  # type: ignore[attr-defined]

    active = read_active_materialization(tmp_path)
    assert active is not None
    assert active.version == LEGACY_VERSION


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    """Running the migration twice must be a no-op on the second pass: no
    additional renames, the pointer is untouched (or refreshed) if
    already at v0."""
    script = _import_script()
    _touch_legacy_table(tmp_path, "doc-A")
    _touch_legacy_table(tmp_path, "_corpus_narrative")

    first = script.migrate(tmp_path)  # type: ignore[attr-defined]
    second = script.migrate(tmp_path)  # type: ignore[attr-defined]

    assert first["renamed"] == 2
    assert second["renamed"] == 0
    assert second["already_versioned"] == 2


def test_migrate_does_not_clobber_post_v0_active_pointer(
    tmp_path: Path,
) -> None:
    """If the operator has already promoted past v0, re-running the
    migration must NOT downgrade the pointer back to v0 — a re-run is a
    legitimate "rerun after a partial failure" scenario and an
    accidental rerun must not regress production."""
    script = _import_script()
    _touch_legacy_table(tmp_path, "doc-A")

    # Pre-write a promoted pointer at a non-v0 version.
    write_active_materialization(
        tmp_path,
        ActiveMaterialization(
            version="a3f1c8b2",
            embed_model_version="voyage:voyage-finance-2:v1",
            promoted_at="2026-05-26T12:00:00Z",
            manifest_count=1,
        ),
    )
    counts = script.migrate(tmp_path)  # type: ignore[attr-defined]
    assert counts["pointer_unchanged"] == 1
    active = read_active_materialization(tmp_path)
    assert active is not None
    assert active.version == "a3f1c8b2"  # not overwritten


def test_migrate_skips_already_versioned_tables(tmp_path: Path) -> None:
    """A table already at `{name}__v0.lance` (e.g., partial prior
    migration) must not be renamed again — that would produce
    `{name}__v0__v0.lance`. The split helper recognizes the suffix and
    skips."""
    script = _import_script()
    (tmp_path / "doc-A__v0.lance").mkdir()

    counts = script.migrate(tmp_path)  # type: ignore[attr-defined]
    assert counts["renamed"] == 0
    assert counts["already_versioned"] == 1
    assert (tmp_path / "doc-A__v0.lance").exists()
    assert not (tmp_path / "doc-A__v0__v0.lance").exists()


def test_migrate_empty_rag_root_is_a_no_op(tmp_path: Path) -> None:
    """The script runs cleanly against an empty rag_root and produces no
    pointer (the cli wrapper checks existence too; this asserts the
    helper itself behaves)."""
    script = _import_script()
    counts = script.migrate(tmp_path)  # type: ignore[attr-defined]
    assert counts == {
        "renamed": 0,
        "already_versioned": 0,
        "pointer_written": 1,  # writes a placeholder pointer with no tables
        "pointer_unchanged": 0,
    }
    # Active pointer has a unknown placeholder embed_model_version.
    active = read_active_materialization(tmp_path)
    assert active is not None
    assert active.manifest_count == 0


def test_migrate_pointer_includes_real_embed_model_version_from_rows(
    tmp_path: Path,
) -> None:
    """When the legacy tables have real rows with the issue #67
    embed_model_version column, the migration's sampled pointer carries
    that value — so an operator constructing a Voyage-matched adapter
    after migration hits the active read path cleanly without the
    mismatch guard firing on a placeholder."""
    import lancedb
    import pyarrow as pa

    from auto_research.extract.embeddings import _schema

    # Build a 1-row LanceDB table at the legacy unversioned path with
    # the row-level version columns issue #67 added.
    db = lancedb.connect(tmp_path)
    schema = _schema(1024)
    rows = [
        {
            "text": "stamped legacy row",
            "vector": [0.0] * 1024,
            "ticker": "NVDA",
            "filing_date": "2025-03-15",
            "fiscal_period": "FY2025",
            "doc_type": "10-K",
            "doc_id": "doc-LEG",
            "parent_id": "doc-LEG:0:18",
            "section_name": "Item 7",
            "chunker_version": "v1",
            "contextual_prompt_version": "v1",
            "embed_model_version": "voyage:voyage-finance-2:v1",
        }
    ]
    db.create_table(
        "doc-LEG",
        data=pa.Table.from_pylist(rows, schema=schema),
        schema=schema,
        mode="overwrite",
    )

    script = _import_script()
    script.migrate(tmp_path)  # type: ignore[attr-defined]

    raw = json.loads((tmp_path / ACTIVE_FILE_NAME).read_text())
    assert raw["embed_model_version"] == "voyage:voyage-finance-2:v1"
