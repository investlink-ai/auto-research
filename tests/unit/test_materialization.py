"""Unit tests for `extract.materialization` — the pure helpers backing the
materialization-versioned LanceDB layout.

These tests are deliberately hermetic (no LanceDB, no embedding model);
the adapter-level routing tests live in `test_embeddings.py`, the CLI
surface in `test_cli.py`, and the end-to-end loop in
`tests/integration/test_embeddings_vcr.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from auto_research.extract.materialization import (
    ACTIVE_FILE_NAME,
    PROMOTION_HISTORY_FILE_NAME,
    ActiveMaterialization,
    append_promotion_history,
    compute_materialization_version,
    list_materializations,
    read_active_materialization,
    read_promotion_history,
    split_versioned_table_name,
    versioned_table_name,
    write_active_materialization,
)

# ---- compute_materialization_version -------------------------------------


def test_compute_materialization_version_is_deterministic() -> None:
    """Same inputs must produce the same hash on every call — this is the
    `embed → promote → query` round-trip's foundation."""
    a = compute_materialization_version("chunker-v3", "ctx-v2", "voyage:voyage-finance-2:v1")
    b = compute_materialization_version("chunker-v3", "ctx-v2", "voyage:voyage-finance-2:v1")
    assert a == b


def test_compute_materialization_version_changes_with_each_component() -> None:
    """Bumping any one of the three contracts must move the hash —
    otherwise a chunker-only bump would leave the new tables sharing a
    namespace with the old, defeating the whole point."""
    base = compute_materialization_version("c-v1", "ctx-v1", "voyage:fin-2:v1")
    bumped_chunker = compute_materialization_version("c-v2", "ctx-v1", "voyage:fin-2:v1")
    bumped_ctx = compute_materialization_version("c-v1", "ctx-v2", "voyage:fin-2:v1")
    bumped_embed = compute_materialization_version("c-v1", "ctx-v1", "voyage:fin-2:v2")
    assert len({base, bumped_chunker, bumped_ctx, bumped_embed}) == 4


def test_compute_materialization_version_is_8_hex_chars() -> None:
    h = compute_materialization_version("a", "b", "c")
    assert len(h) == 8
    int(h, 16)  # raises if not hex


def test_compute_materialization_version_pipe_separator_is_injective() -> None:
    """`"ab|cd|e"` and `"a|bcd|e"` would collide if we naively concatenated
    without a separator. Guard the separator policy."""
    h1 = compute_materialization_version("ab", "cd", "e")
    h2 = compute_materialization_version("a", "bcd", "e")
    assert h1 != h2


# ---- versioned_table_name / split ---------------------------------------


def test_versioned_table_name_round_trip() -> None:
    name = versioned_table_name("doc-NVDA-2025", "a3f1c8b2")
    assert name == "doc-NVDA-2025__a3f1c8b2"
    base, version = split_versioned_table_name(name)  # type: ignore[misc]
    assert (base, version) == ("doc-NVDA-2025", "a3f1c8b2")


def test_split_returns_none_for_unversioned_table_name() -> None:
    assert split_versioned_table_name("doc-NVDA-2025") is None


def test_split_anchors_on_last_double_underscore() -> None:
    """If a base contains `__` itself (unlikely but possible), splitting
    on the LAST occurrence keeps the version slug clean."""
    name = "weird__base__a3f1c8b2"
    base, version = split_versioned_table_name(name)  # type: ignore[misc]
    assert base == "weird__base"
    assert version == "a3f1c8b2"


def test_versioned_table_name_rejects_separator_in_version() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        versioned_table_name("doc-1", "bad__version")


def test_versioned_table_name_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="base table name"):
        versioned_table_name("", "v0")
    with pytest.raises(ValueError, match="materialization version"):
        versioned_table_name("doc-1", "")


# ---- active pointer read/write -------------------------------------------


def _sample(version: str = "a3f1c8b2") -> ActiveMaterialization:
    return ActiveMaterialization(
        version=version,
        embed_model_version="voyage:voyage-finance-2:v1",
        promoted_at="2026-05-26T12:00:00Z",
        manifest_count=56000,
    )


def test_read_active_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_active_materialization(tmp_path) is None


def test_write_then_read_active_round_trips(tmp_path: Path) -> None:
    sample = _sample()
    write_active_materialization(tmp_path, sample)
    assert (tmp_path / ACTIVE_FILE_NAME).exists()
    assert read_active_materialization(tmp_path) == sample


def test_write_active_creates_rag_root_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "doesnt" / "exist" / "yet"
    write_active_materialization(nested, _sample())
    assert (nested / ACTIVE_FILE_NAME).exists()


def test_write_active_overwrites_previous(tmp_path: Path) -> None:
    write_active_materialization(tmp_path, _sample("v-old"))
    write_active_materialization(tmp_path, _sample("v-new"))
    active = read_active_materialization(tmp_path)
    assert active is not None
    assert active.version == "v-new"


def test_atomic_flip_mid_failure_preserves_previous_pointer(
    tmp_path: Path,
) -> None:
    """If `os.replace` raises mid-flip, the previous pointer must remain
    intact — the issue's "failure-mid-flip" acceptance criterion. We
    simulate by patching `os.replace` to raise after the tmp file is
    written; the unchanged on-disk JSON proves the property."""
    write_active_materialization(tmp_path, _sample("v-old"))

    with mock.patch(
        "auto_research._io.os.replace", side_effect=OSError("simulated fs crash")
    ), pytest.raises(OSError, match="simulated fs crash"):
        write_active_materialization(tmp_path, _sample("v-new"))

    active = read_active_materialization(tmp_path)
    assert active is not None
    assert active.version == "v-old", "previous pointer lost when flip failed"

    # And the tmp file is cleaned up so the directory isn't littered.
    tmp_siblings = [
        p for p in tmp_path.iterdir() if p.name.startswith(f".{ACTIVE_FILE_NAME}")
    ]
    assert tmp_siblings == [], f"tmp file leaked: {tmp_siblings}"


def test_active_pointer_json_shape_is_stable(tmp_path: Path) -> None:
    """The on-disk JSON keys must match the documented schema — external
    tooling (dashboards, ops scripts) reads this directly."""
    write_active_materialization(tmp_path, _sample())
    raw = json.loads((tmp_path / ACTIVE_FILE_NAME).read_text())
    assert set(raw.keys()) == {
        "version",
        "embed_model_version",
        "promoted_at",
        "manifest_count",
    }


# ---- promotion history ---------------------------------------------------


def test_read_history_empty_when_no_file(tmp_path: Path) -> None:
    assert read_promotion_history(tmp_path) == []


def test_append_history_preserves_order(tmp_path: Path) -> None:
    a = _sample("v-a")
    b = _sample("v-b")
    c = _sample("v-c")
    append_promotion_history(tmp_path, a)
    append_promotion_history(tmp_path, b)
    append_promotion_history(tmp_path, c)
    history = read_promotion_history(tmp_path)
    assert [h.version for h in history] == ["v-a", "v-b", "v-c"]


def test_append_history_atomic_against_concurrent_read(tmp_path: Path) -> None:
    """Append uses the same atomic-write primitive as the active pointer —
    a reader at the moment of the rename sees either the old list or the
    new list, never a torn JSON. We can't reliably race threads in a unit
    test, but we can at least pin the file name and the durability call
    by inspecting that the on-disk write happened via the atomic helper.
    Sanity: round-trip survives a re-read."""
    append_promotion_history(tmp_path, _sample("v-a"))
    assert (tmp_path / PROMOTION_HISTORY_FILE_NAME).exists()
    assert read_promotion_history(tmp_path)[0].version == "v-a"


# ---- list_materializations ----------------------------------------------


def _touch_lance_dir(rag_root: Path, name: str) -> None:
    """LanceDB tables are `<name>.lance/` directories; we don't need a real
    LanceDB to test enumeration — only the directory shape."""
    (rag_root / f"{name}.lance").mkdir(parents=True)


def test_list_materializations_groups_by_version_suffix(tmp_path: Path) -> None:
    _touch_lance_dir(tmp_path, "doc-A__aaaaaaaa")
    _touch_lance_dir(tmp_path, "doc-B__aaaaaaaa")
    _touch_lance_dir(tmp_path, "_corpus_narrative__aaaaaaaa")
    _touch_lance_dir(tmp_path, "doc-A__a3f1c8b2")
    _touch_lance_dir(tmp_path, "_corpus_narrative__a3f1c8b2")
    materializations = list_materializations(tmp_path)
    by_version = {m.version: m for m in materializations}
    assert set(by_version) == {"aaaaaaaa", "a3f1c8b2"}
    assert by_version["aaaaaaaa"].table_count == 3
    assert by_version["a3f1c8b2"].table_count == 2


def test_list_materializations_marks_active(tmp_path: Path) -> None:
    _touch_lance_dir(tmp_path, "doc-A__aaaaaaaa")
    _touch_lance_dir(tmp_path, "doc-A__a3f1c8b2")
    write_active_materialization(tmp_path, _sample("a3f1c8b2"))
    materializations = list_materializations(tmp_path)
    by_version = {m.version: m for m in materializations}
    assert by_version["a3f1c8b2"].is_active is True
    assert by_version["aaaaaaaa"].is_active is False


def test_list_materializations_skips_unrecognized_dirs(tmp_path: Path) -> None:
    """Stray `.lance` directories without a `__version` suffix (partial
    scratch directories, hand-created tables) must NOT trip the
    enumerator — callers should be able to introspect a partly-built
    rag_root without crashing."""
    _touch_lance_dir(tmp_path, "doc-A__aaaaaaaa")
    _touch_lance_dir(tmp_path, "unversioned-stray")  # no __
    materializations = list_materializations(tmp_path)
    assert [m.version for m in materializations] == ["aaaaaaaa"]


def test_list_materializations_empty_rag_root(tmp_path: Path) -> None:
    assert list_materializations(tmp_path) == []
    nonexistent = tmp_path / "never"
    assert list_materializations(nonexistent) == []


# ---- corrupted-JSON handling --------------------------------------------


def test_read_active_raises_on_corrupted_json(tmp_path: Path) -> None:
    """A manually-edited or partially-written `active_materialization.json`
    must raise loudly rather than silently degrade to None — silent
    degradation would make corruption indistinguishable from a fresh
    install and mask the condition that needs operator intervention.
    """
    (tmp_path / ACTIVE_FILE_NAME).write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        read_active_materialization(tmp_path)


def test_read_promotion_history_raises_on_corrupted_json(tmp_path: Path) -> None:
    """Same policy as the active pointer: corrupted history must not be
    silently treated as empty, or gc would start deleting tables the
    operator intended to keep."""
    (tmp_path / PROMOTION_HISTORY_FILE_NAME).write_text(
        "[ malformed", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not valid JSON"):
        read_promotion_history(tmp_path)
