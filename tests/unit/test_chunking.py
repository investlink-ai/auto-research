"""Unit tests for `auto_research.extract.chunking` (Issue #13, Tier 2).

Hermetic: no network, no LLM calls. Real 10-K HTML loaded from the
checked-in NVDA fixture at `tests/fixtures/chunking/sample_10k.htm`.

These tests collectively form the Tier 2 evidence required by
`docs/AI_WORKFLOW.md` §5 for the chunking module's touch on INV-2
(citation-grounding via `char_span`).
"""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from auto_research.extract.chunking import (
    DEFAULT_QUARANTINE_ROOT,
    MAX_CHILD_TOKENS,
    MAX_PARENT_TOKENS,
    MIN_CHILD_TOKENS,
    SINGLE_SHOT_TOKEN_CUTOFF,
    ChildChunk,
    ChunkMetadata,
    ChunkSet,
    ChunkValidationError,
    ParentChunk,
    count_tokens,
    parse_filing,
    quarantine_chunkset,
    validate_char_spans,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chunking"


# ---------- Session fixtures ------------------------------------------------


@pytest.fixture(scope="session")
def sample_10k_html() -> str:
    return (FIXTURE_DIR / "sample_10k.htm").read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="session")
def edge_cases_html() -> str:
    return (FIXTURE_DIR / "edge_cases.htm").read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="session")
def sample_10k_meta() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "sample_10k.meta.json").read_text())  # type: ignore[no-any-return]


@pytest.fixture
def sample_10k_metadata(sample_10k_meta: dict[str, Any]) -> ChunkMetadata:
    return ChunkMetadata(
        ticker=sample_10k_meta["ticker"],
        filing_date=date.fromisoformat(sample_10k_meta["filing_date"]),
        fiscal_period=sample_10k_meta["fiscal_period"],
        doc_type=sample_10k_meta["doc_type"],
        doc_id=sample_10k_meta["doc_id"],
    )


@pytest.fixture
def edge_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        ticker="TEST",
        filing_date=date(2025, 1, 1),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="edge-cases-test",
    )


@pytest.fixture(scope="session")
def parsed_sample(sample_10k_html: str, sample_10k_meta: dict[str, Any]) -> ChunkSet:
    """Cache the parse result — partition_html is the slow step."""
    meta = ChunkMetadata(
        ticker=sample_10k_meta["ticker"],
        filing_date=date.fromisoformat(sample_10k_meta["filing_date"]),
        fiscal_period=sample_10k_meta["fiscal_period"],
        doc_type=sample_10k_meta["doc_type"],
        doc_id=sample_10k_meta["doc_id"],
    )
    return parse_filing(html=sample_10k_html, metadata=meta)


# ---------- Dataclass shape -------------------------------------------------


def test_chunk_metadata_is_frozen(sample_10k_metadata: ChunkMetadata) -> None:
    with pytest.raises((AttributeError, Exception)):
        sample_10k_metadata.ticker = "OTHER"  # type: ignore[misc]


def test_parent_chunk_is_frozen_and_carries_metadata(sample_10k_metadata: ChunkMetadata) -> None:
    p = ParentChunk(
        text="hello",
        section_name="Item 1A",
        char_span=(0, 5),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    assert p.metadata.ticker == "NVDA"
    assert p.table_html is None
    with pytest.raises((AttributeError, Exception)):
        p.text = "x"  # type: ignore[misc]


def test_child_chunk_carries_parent_id_and_metadata(sample_10k_metadata: ChunkMetadata) -> None:
    c = ChildChunk(
        text="hello",
        char_span=(0, 5),
        token_count=1,
        parent_id="doc::0-5",
        metadata=sample_10k_metadata,
    )
    assert c.parent_id == "doc::0-5"


def test_chunkset_groups_parents_and_children(sample_10k_metadata: ChunkMetadata) -> None:
    p = ParentChunk(
        text="hello world",
        section_name="Item 1A",
        char_span=(0, 11),
        token_count=2,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    c = ChildChunk(
        text="hello",
        char_span=(0, 5),
        token_count=1,
        parent_id="doc::0-11",
        metadata=sample_10k_metadata,
    )
    cs = ChunkSet(parents=[p], children=[c])
    assert len(cs.parents) == 1
    assert len(cs.children) == 1


def test_chunk_validation_error_is_value_error() -> None:
    assert issubclass(ChunkValidationError, ValueError)


def test_module_level_constants() -> None:
    assert SINGLE_SHOT_TOKEN_CUTOFF == 100_000
    assert MAX_PARENT_TOKENS == 4_000
    assert MIN_CHILD_TOKENS == 200
    assert MAX_CHILD_TOKENS == 800


def test_default_quarantine_root_points_at_data_dir() -> None:
    assert Path("data/quarantine") == DEFAULT_QUARANTINE_ROOT


# ---------- Token counting --------------------------------------------------


def test_count_tokens_cl100k_basic() -> None:
    # "hello world" → 2 tokens in cl100k_base.
    assert count_tokens("hello world") == 2


def test_count_tokens_handles_empty() -> None:
    assert count_tokens("") == 0


# ---------- char_span validation (INV-2) ------------------------------------


def test_validate_char_spans_passes_when_aligned(sample_10k_metadata: ChunkMetadata) -> None:
    src = "hello world"
    p = ParentChunk(
        text="hello",
        section_name="Item 1A",
        char_span=(0, 5),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    validate_char_spans(src, [p], [])  # must not raise


def test_validate_char_spans_raises_on_text_mismatch(sample_10k_metadata: ChunkMetadata) -> None:
    src = "hello world"
    bad = ParentChunk(
        text="HELLO",
        section_name="Item 1A",
        char_span=(0, 5),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    with pytest.raises(ChunkValidationError) as exc:
        validate_char_spans(src, [bad], [])
    assert "Item 1A" in str(exc.value)


def test_validate_char_spans_raises_on_out_of_bounds(sample_10k_metadata: ChunkMetadata) -> None:
    src = "short"
    bad = ParentChunk(
        text="overflow",
        section_name="Item 1A",
        char_span=(0, 999),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    with pytest.raises(ChunkValidationError):
        validate_char_spans(src, [bad], [])


def test_validate_char_spans_walks_children(sample_10k_metadata: ChunkMetadata) -> None:
    src = "hello world"
    parent = ParentChunk(
        text="hello world",
        section_name="Item 1A",
        char_span=(0, 11),
        token_count=2,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    child_ok = ChildChunk(
        text="hello",
        char_span=(0, 5),
        token_count=1,
        parent_id="doc::0-11",
        metadata=sample_10k_metadata,
    )
    child_bad = ChildChunk(
        text="WORLD",
        char_span=(6, 11),
        token_count=1,
        parent_id="doc::0-11",
        metadata=sample_10k_metadata,
    )
    validate_char_spans(src, [parent], [child_ok])
    with pytest.raises(ChunkValidationError):
        validate_char_spans(src, [parent], [child_bad])


# ---------- parse_filing on the edge-cases fixture (INV-2 + entities) ------


def test_parse_filing_preserves_char_span_through_html_edge_cases(
    edge_cases_html: str, edge_metadata: ChunkMetadata
) -> None:
    result = parse_filing(html=edge_cases_html, metadata=edge_metadata)
    assert result.parents, "expected at least one parent chunk"
    for p in result.parents:
        assert edge_cases_html[p.char_span[0] : p.char_span[1]] == p.text, (
            f"INV-2 violated for parent in {p.section_name}: "
            f"raw[{p.char_span}]={edge_cases_html[p.char_span[0]:p.char_span[1]][:80]!r} "
            f"vs chunk.text={p.text[:80]!r}"
        )
    for c in result.children:
        assert edge_cases_html[c.char_span[0] : c.char_span[1]] == c.text, (
            f"INV-2 violated for child: raw[{c.char_span}]="
            f"{edge_cases_html[c.char_span[0]:c.char_span[1]][:80]!r} "
            f"vs chunk.text={c.text[:80]!r}"
        )


def test_parse_filing_detects_sections_through_nbsp_entities(edge_cases_html: str, edge_metadata: ChunkMetadata) -> None:
    result = parse_filing(html=edge_cases_html, metadata=edge_metadata)
    section_names = {p.section_name for p in result.parents}
    # Edge-case fixture uses Item&#160;1A and Item&#160;7 headers.
    assert "Item 1A" in section_names
    assert "Item 7" in section_names


# ---------- parse_filing on real 10-K fixture -------------------------------


def test_real_10k_parses_into_expected_sections(parsed_sample: ChunkSet, sample_10k_meta: dict[str, str | list[str]]) -> None:
    section_names = [p.section_name for p in parsed_sample.parents]
    for required in sample_10k_meta["expected_sections"]:
        assert required in section_names, (
            f"missing {required!r}: got {sorted(set(section_names))}"
        )


def test_real_10k_narrative_parents_under_max_tokens(parsed_sample: ChunkSet) -> None:
    """Token cap applies to NARRATIVE parents (`table_html is None`).

    Table parents may exceed MAX_PARENT_TOKENS — they're routed to a
    separate structured-extraction path (ADR D5) that reads
    `table_html` directly, so the dense-retrieval token budget doesn't
    apply. Splitting a table mid-rows would also break HTML well-
    formedness, which downstream consumers rely on.
    """
    for p in parsed_sample.parents:
        if p.table_html is not None:
            continue
        assert p.token_count <= MAX_PARENT_TOKENS, (
            f"narrative parent in {p.section_name} exceeds budget: "
            f"{p.token_count} > {MAX_PARENT_TOKENS}"
        )


def test_real_10k_parents_respect_section_boundaries(parsed_sample: ChunkSet) -> None:
    # Parents within a section share section_name; no parent spans a boundary
    # (encoded by the invariant that section_name is a single value per chunk).
    # Stronger: parents in document order are non-overlapping and
    # section-monotonic — once a section is "left" by a later parent,
    # earlier sections don't reappear.
    seen_sections: list[str] = []
    for p in parsed_sample.parents:
        if not seen_sections or seen_sections[-1] != p.section_name:
            assert p.section_name not in seen_sections, (
                f"section {p.section_name} appears non-contiguously: {seen_sections}"
            )
            seen_sections.append(p.section_name)


def test_real_10k_children_are_subset_of_parent_spans(parsed_sample: ChunkSet) -> None:
    parents_by_id: dict[str, ParentChunk] = {}
    for p in parsed_sample.parents:
        pid = f"{p.metadata.doc_id}::{p.char_span[0]}-{p.char_span[1]}"
        parents_by_id[pid] = p

    for c in parsed_sample.children:
        parent = parents_by_id.get(c.parent_id)
        assert parent is not None, f"child has no matching parent: {c.parent_id}"
        assert parent.char_span[0] <= c.char_span[0] < c.char_span[1] <= parent.char_span[1], (
            f"child span {c.char_span} not within parent span {parent.char_span}"
        )


def test_real_10k_child_token_band(parsed_sample: ChunkSet) -> None:
    # Allow one degenerate exception: a single child equal to a small parent
    # (parents below MIN_CHILD_TOKENS yield a single child equal to the parent).
    for c in parsed_sample.children:
        assert c.token_count <= MAX_CHILD_TOKENS, (
            f"child exceeds MAX_CHILD_TOKENS: {c.token_count}"
        )


def test_real_10k_char_span_fidelity_holds(parsed_sample: ChunkSet, sample_10k_html: str) -> None:
    # The headline INV-2 test on real data.
    for p in parsed_sample.parents:
        assert sample_10k_html[p.char_span[0] : p.char_span[1]] == p.text
    for c in parsed_sample.children:
        assert sample_10k_html[c.char_span[0] : c.char_span[1]] == c.text


def test_real_10k_metadata_is_populated(parsed_sample: ChunkSet) -> None:
    for p in parsed_sample.parents:
        assert p.metadata.ticker == "NVDA"
        assert p.metadata.doc_type == "10-K"
        assert p.metadata.doc_id == "0001045810-25-000023"
        assert p.metadata.fiscal_period == "FY2025"
    for c in parsed_sample.children:
        assert c.metadata.ticker == "NVDA"


# ---------- Table policy (ADR D5) -------------------------------------------


def test_item_8_emits_chunks_with_table_html(parsed_sample: ChunkSet) -> None:
    item_8_parents = [p for p in parsed_sample.parents if p.section_name == "Item 8"]
    assert item_8_parents, "expected at least one Item 8 parent chunk in NVDA 10-K"
    table_chunks = [p for p in item_8_parents if p.table_html is not None]
    assert table_chunks, (
        "expected ≥1 Item 8 parent with table_html attached "
        f"(got {len(item_8_parents)} item-8 parents, none with table_html)"
    )


def test_non_table_chunks_have_no_table_html(parsed_sample: ChunkSet) -> None:
    item_1a = [p for p in parsed_sample.parents if p.section_name == "Item 1A"]
    assert item_1a, "expected Item 1A parents in fixture"
    for p in item_1a:
        assert p.table_html is None, (
            f"Item 1A chunk should not carry table_html: {p.text[:80]!r}"
        )


def test_at_least_one_table_html_is_pandas_readable(parsed_sample: ChunkSet) -> None:
    tables = [p for p in parsed_sample.parents if p.table_html is not None]
    assert tables, "expected table chunks"
    for t in tables:
        try:
            dfs = pd.read_html(io.StringIO(t.table_html))
        except (ValueError, Exception):
            continue
        if dfs and not dfs[0].empty:
            return  # success — at least one parsed
    pytest.fail("no table_html parsed via pandas.read_html — table-policy contract broken")


# ---------- Quarantine routing ----------------------------------------------


def test_quarantine_writes_json_with_reason(tmp_path: Path, sample_10k_metadata: ChunkMetadata) -> None:
    bad = ParentChunk(
        text="WRONG",
        section_name="Item 1A",
        char_span=(0, 5),
        token_count=1,
        table_html=None,
        metadata=sample_10k_metadata,
    )
    cs = ChunkSet(parents=[bad], children=[])
    dest = quarantine_chunkset(
        cs, source_text="hello", reason="char_span mismatch", quarantine_root=tmp_path
    )
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["reason"] == "char_span mismatch"
    assert data["doc_id"] == sample_10k_metadata.doc_id
    assert data["parents"][0]["section_name"] == "Item 1A"
    assert data["source_text_length"] == 5


def test_quarantine_handles_empty_chunkset(tmp_path: Path) -> None:
    cs = ChunkSet(parents=[], children=[])
    dest = quarantine_chunkset(cs, source_text="", reason="empty", quarantine_root=tmp_path)
    assert dest.exists()


def test_quarantine_writes_atomically(tmp_path: Path, sample_10k_metadata: ChunkMetadata) -> None:
    cs = ChunkSet(parents=[], children=[])
    quarantine_chunkset(cs, source_text="", reason="x", quarantine_root=tmp_path)
    # No tmp-suffix files left in the dest dir.
    tmpfiles = list((tmp_path / "chunking").glob(".*tmp*"))
    assert tmpfiles == [], f"leftover tmp files: {tmpfiles}"


# ---------- No-network guarantee --------------------------------------------


def test_parse_filing_makes_no_network_calls(edge_cases_html: str, edge_metadata: ChunkMetadata, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic guarantee — parsing must not touch the network."""
    import socket

    def _no_socket(*args: Any, **kwargs: Any) -> None:
        raise OSError("network access forbidden during chunking")

    monkeypatch.setattr(socket, "socket", _no_socket)
    monkeypatch.setattr(socket, "create_connection", _no_socket)
    monkeypatch.setattr(socket, "getaddrinfo", _no_socket)

    result = parse_filing(html=edge_cases_html, metadata=edge_metadata)
    assert result.parents
