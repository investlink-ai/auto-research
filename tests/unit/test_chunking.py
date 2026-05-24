"""Unit tests for `auto_research.extract.chunking` (Issue #13, Tier 2).

Hermetic: no network, no LLM calls. Real 10-K HTML loaded from the
checked-in NVDA fixture at `tests/fixtures/chunking/sample_10k.htm`.

These tests collectively form the Tier 2 evidence required by
`docs/AI_WORKFLOW.md` §5 for the chunking module's touch on INV-2
(citation-grounding via `char_span`).
"""

from __future__ import annotations

import dataclasses
import inspect
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from auto_research.extract import chunking as chunking_mod
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
    validate_or_quarantine_chunkset,
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
    """Cache the parse result — section detection is the slow step."""
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
    with pytest.raises(dataclasses.FrozenInstanceError):
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
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.text = "x"  # type: ignore[misc]


def test_child_chunk_carries_required_fields(sample_10k_metadata: ChunkMetadata) -> None:
    c = ChildChunk(
        text="hello",
        char_span=(0, 5),
        token_count=1,
        parent_id="doc::0-5",
        section_name="Item 1A",
        from_table=False,
        metadata=sample_10k_metadata,
    )
    assert c.parent_id == "doc::0-5"
    assert c.section_name == "Item 1A"  # ADR D7 contract
    assert c.from_table is False  # ADR D5 contract


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
        section_name="Item 1A",
        from_table=False,
        metadata=sample_10k_metadata,
    )
    cs = ChunkSet(parents=(p,), children=(c,))
    assert len(cs.parents) == 1
    assert len(cs.children) == 1


def test_chunkset_fields_are_tuples_not_lists(parsed_sample: ChunkSet) -> None:
    """ADR/code-review P1-12: frozen ChunkSet uses tuples so consumers
    cannot mutate the parents/children sequence in place."""
    assert isinstance(parsed_sample.parents, tuple)
    assert isinstance(parsed_sample.children, tuple)
    with pytest.raises(AttributeError):
        parsed_sample.parents.append(parsed_sample.parents[0])  # type: ignore[attr-defined]


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


def _mk_parent(meta: ChunkMetadata, **overrides: Any) -> ParentChunk:
    defaults: dict[str, Any] = {
        "text": "hello",
        "section_name": "Item 1A",
        "char_span": (0, 5),
        "token_count": 1,
        "table_html": None,
        "metadata": meta,
    }
    defaults.update(overrides)
    return ParentChunk(**defaults)


def _mk_child(meta: ChunkMetadata, **overrides: Any) -> ChildChunk:
    defaults: dict[str, Any] = {
        "text": "hello",
        "char_span": (0, 5),
        "token_count": 1,
        "parent_id": "doc::0-11",
        "section_name": "Item 1A",
        "from_table": False,
        "metadata": meta,
    }
    defaults.update(overrides)
    return ChildChunk(**defaults)


def test_validate_char_spans_passes_when_aligned(sample_10k_metadata: ChunkMetadata) -> None:
    validate_char_spans("hello world", [_mk_parent(sample_10k_metadata)], [])


def test_validate_char_spans_raises_on_text_mismatch(sample_10k_metadata: ChunkMetadata) -> None:
    bad = _mk_parent(sample_10k_metadata, text="HELLO")
    with pytest.raises(ChunkValidationError) as exc:
        validate_char_spans("hello world", [bad], [])
    assert "Item 1A" in str(exc.value)


def test_validate_char_spans_raises_on_out_of_bounds(sample_10k_metadata: ChunkMetadata) -> None:
    bad = _mk_parent(sample_10k_metadata, text="overflow", char_span=(0, 999))
    with pytest.raises(ChunkValidationError):
        validate_char_spans("short", [bad], [])


def test_validate_char_spans_walks_children(sample_10k_metadata: ChunkMetadata) -> None:
    parent = _mk_parent(sample_10k_metadata, text="hello world", char_span=(0, 11), token_count=2)
    child_ok = _mk_child(sample_10k_metadata, text="hello", char_span=(0, 5))
    child_bad = _mk_child(sample_10k_metadata, text="WORLD", char_span=(6, 11))
    validate_char_spans("hello world", [parent], [child_ok])
    with pytest.raises(ChunkValidationError):
        validate_char_spans("hello world", [parent], [child_bad])


# ---------- parse_filing on the edge-cases fixture (INV-2 + entities) ------


def test_parse_filing_preserves_char_span_through_html_edge_cases(
    edge_cases_html: str, edge_metadata: ChunkMetadata
) -> None:
    result = parse_filing(html=edge_cases_html, metadata=edge_metadata)
    assert result.parents, "expected at least one parent chunk"
    for p in result.parents:
        assert edge_cases_html[p.char_span[0] : p.char_span[1]] == p.text, (
            f"INV-2 violated for parent in {p.section_name}"
        )
    for c in result.children:
        assert edge_cases_html[c.char_span[0] : c.char_span[1]] == c.text, (
            f"INV-2 violated for child in {c.section_name}"
        )


def test_parse_filing_detects_sections_through_nbsp_entities(
    edge_cases_html: str, edge_metadata: ChunkMetadata
) -> None:
    result = parse_filing(html=edge_cases_html, metadata=edge_metadata)
    section_names = {p.section_name for p in result.parents}
    assert "Item 1A" in section_names
    assert "Item 7" in section_names


def test_uppercase_entity_name_does_not_match_as_header(
    edge_metadata: ChunkMetadata,
) -> None:
    """P1-11: HTML5 entity names are case-sensitive (`&nbsp;` valid,
    `&NBSP;` is not). The Item-header regex must not over-match."""
    # `Item&NBSP;5` should NOT be detected as Item 5 (uppercase entity is
    # invalid and browsers render it as literal text). The fixture below
    # has no real headers, so detection should fall back to a single
    # "Body" section.
    html = (
        "<html><body>"
        "<p>Filler prose " + ("text " * 200) + "</p>"
        "<p>Some context Item&NBSP;5 inline reference " + ("more " * 200) + "</p>"
        "</body></html>"
    )
    result = parse_filing(html=html, metadata=edge_metadata)
    section_names = {p.section_name for p in result.parents}
    assert "Item 5" not in section_names, (
        "uppercase HTML entity name should not match as a section header"
    )


def test_inline_span_item_reference_does_not_false_positive(
    edge_metadata: ChunkMetadata,
) -> None:
    """P1-9: An Item-N mention inside an inline `<span>` should NOT be
    treated as a structural section header. The preceding tag is
    inline, not block-level."""
    # Build a doc where Item 7 appears only inside a `<span>` inside
    # running prose, never as a structural header.
    html = (
        "<html><body><h1>Item 1A. Risk Factors</h1>"
        + ("<p>Risk factor body content " + ("foo " * 200) + "</p>") * 3
        + "<p>... compared to <span>Item 7</span> discussed elsewhere "
        + ("baz " * 200)
        + "</p>"
        "</body></html>"
    )
    result = parse_filing(html=html, metadata=edge_metadata)
    section_names = {p.section_name for p in result.parents}
    assert "Item 7" not in section_names, (
        f"inline `<span>Item 7</span>` should not register as section: {section_names}"
    )


# ---------- parse_filing on real 10-K fixture -------------------------------


def test_real_10k_parses_into_expected_sections(
    parsed_sample: ChunkSet, sample_10k_meta: dict[str, Any]
) -> None:
    section_names = [p.section_name for p in parsed_sample.parents]
    for required in sample_10k_meta["expected_sections"]:
        assert required in section_names, (
            f"missing {required!r}: got {sorted(set(section_names))}"
        )


def test_real_10k_narrative_parents_under_max_tokens(parsed_sample: ChunkSet) -> None:
    """Token cap applies to NARRATIVE parents (`table_html is None`).

    Table parents may exceed MAX_PARENT_TOKENS — they take a separate
    structured-extraction path (ADR D5) and splitting them mid-row would
    break HTML well-formedness.
    """
    for p in parsed_sample.parents:
        if p.table_html is not None:
            continue
        assert p.token_count <= MAX_PARENT_TOKENS, (
            f"narrative parent in {p.section_name} exceeds budget: "
            f"{p.token_count} > {MAX_PARENT_TOKENS}"
        )


def test_real_10k_parents_respect_section_boundaries(parsed_sample: ChunkSet) -> None:
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
        assert parent.char_span[0] <= c.char_span[0] < c.char_span[1] <= parent.char_span[1]


def test_real_10k_child_token_band_narrative_only(parsed_sample: ChunkSet) -> None:
    """Narrative children stay within MAX_CHILD_TOKENS. Table children
    can exceed it (they equal their parent per ADR D5; that's the
    documented degenerate case)."""
    for c in parsed_sample.children:
        if c.from_table:
            continue
        assert c.token_count <= MAX_CHILD_TOKENS, (
            f"narrative child exceeds MAX_CHILD_TOKENS: {c.token_count}"
        )


def test_real_10k_child_section_name_matches_parent(parsed_sample: ChunkSet) -> None:
    """ADR D7: child carries section_name so LanceDB can filter at
    index time without a parent JOIN."""
    parents_by_id: dict[str, ParentChunk] = {}
    for p in parsed_sample.parents:
        pid = f"{p.metadata.doc_id}::{p.char_span[0]}-{p.char_span[1]}"
        parents_by_id[pid] = p

    for c in parsed_sample.children:
        parent = parents_by_id[c.parent_id]
        assert c.section_name == parent.section_name


def test_real_10k_char_span_fidelity_holds(
    parsed_sample: ChunkSet, sample_10k_html: str
) -> None:
    for p in parsed_sample.parents:
        assert sample_10k_html[p.char_span[0] : p.char_span[1]] == p.text
    for c in parsed_sample.children:
        assert sample_10k_html[c.char_span[0] : c.char_span[1]] == c.text


def test_real_10k_metadata_is_populated(parsed_sample: ChunkSet) -> None:
    for p in parsed_sample.parents:
        assert p.metadata.ticker == "NVDA"
        assert p.metadata.doc_type == "10-K"
        assert p.metadata.doc_id == "0001045810-25-000023"
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
    assert item_1a
    for p in item_1a:
        assert p.table_html is None


def test_at_least_one_table_html_is_pandas_readable(parsed_sample: ChunkSet) -> None:
    tables = [p for p in parsed_sample.parents if p.table_html is not None]
    assert tables
    for t in tables:
        try:
            dfs = pd.read_html(io.StringIO(t.table_html))
        except ValueError:
            continue
        if dfs and not dfs[0].empty:
            return  # success
    pytest.fail("no table_html parsed via pandas.read_html — table-policy contract broken")


def test_table_parents_subdivide_to_single_child_equal_to_parent(
    parsed_sample: ChunkSet,
) -> None:
    """ADR D5: table parents are atomic. Splitting at `</td>` would
    produce fragments without `</table>`. Each table parent must yield
    exactly one child with from_table=True and char_span equal to
    parent's char_span."""
    children_by_parent_id: dict[str, list[ChildChunk]] = {}
    for c in parsed_sample.children:
        children_by_parent_id.setdefault(c.parent_id, []).append(c)

    for p in parsed_sample.parents:
        if p.table_html is None:
            continue
        pid = f"{p.metadata.doc_id}::{p.char_span[0]}-{p.char_span[1]}"
        kids = children_by_parent_id.get(pid, [])
        assert len(kids) == 1, (
            f"table parent in {p.section_name} should yield 1 child, got {len(kids)}"
        )
        kid = kids[0]
        assert kid.from_table is True
        assert kid.char_span == p.char_span
        assert kid.text == p.text


def test_narrative_children_have_from_table_false(parsed_sample: ChunkSet) -> None:
    for c in parsed_sample.children:
        if c.from_table:
            continue
        # The corresponding parent must also be a narrative parent.
        pid_parts = c.parent_id.split("::")[-1].split("-")
        parent_span = (int(pid_parts[0]), int(pid_parts[1]))
        matching_parents = [p for p in parsed_sample.parents if p.char_span == parent_span]
        assert matching_parents
        assert matching_parents[0].table_html is None


# ---------- subdivide_to_children — quarantine-or-raise on bad input -------


def test_subdivide_raises_when_no_safe_split_exists(
    edge_metadata: ChunkMetadata,
) -> None:
    """P1-7: a narrative parent containing a single span whose tokens
    exceed MAX_CHILD_TOKENS with no internal break boundary must raise
    ChunkValidationError (caller routes to quarantine), not silently
    emit an oversized child."""
    # Build a parent whose text has no `. ` / `</p>` boundaries and
    # exceeds MAX_CHILD_TOKENS in a single span. Use a diverse char
    # mix so tiktoken cannot compress repeated patterns into a small
    # token count.
    import string

    rng = list(string.ascii_letters + string.digits)
    # 12_000 chars of cycled non-repeating bytes — comfortably > 800
    # tokens without sentence punctuation that would create boundaries.
    long_run = "".join(rng[i % len(rng)] for i in range(12_000))
    parent = ParentChunk(
        text=long_run,
        section_name="Item 1A",
        char_span=(0, len(long_run)),
        token_count=count_tokens(long_run),
        table_html=None,
        metadata=edge_metadata,
    )
    assert parent.token_count > MAX_CHILD_TOKENS, parent.token_count
    with pytest.raises(ChunkValidationError):
        chunking_mod.subdivide_to_children(parent)


# ---------- Quarantine routing ----------------------------------------------


def test_quarantine_writes_json_with_reason(
    tmp_path: Path, sample_10k_metadata: ChunkMetadata
) -> None:
    bad = _mk_parent(sample_10k_metadata, text="WRONG")
    cs = ChunkSet(parents=(bad,), children=())
    dest = quarantine_chunkset(
        cs,
        doc_id=sample_10k_metadata.doc_id,
        source_text="hello",
        reason="char_span mismatch",
        quarantine_root=tmp_path,
    )
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["reason"] == "char_span mismatch"
    assert data["doc_id"] == sample_10k_metadata.doc_id
    assert data["parents"][0]["section_name"] == "Item 1A"
    assert data["source_text_length"] == 5


def test_quarantine_requires_doc_id(tmp_path: Path) -> None:
    """P0-4: quarantine_chunkset requires an explicit, non-empty doc_id.

    Previously, empty ChunkSets resolved to the literal string 'empty'
    and collided at chunking/empty.json across calls. The fix forces
    the caller to provide doc_id verbatim from the document context.
    """
    empty_cs = ChunkSet(parents=(), children=())
    with pytest.raises(ValueError, match="doc_id"):
        quarantine_chunkset(
            empty_cs,
            doc_id="",
            source_text="",
            reason="empty",
            quarantine_root=tmp_path,
        )
    with pytest.raises(ValueError, match="doc_id"):
        quarantine_chunkset(
            empty_cs,
            doc_id="   ",
            source_text="",
            reason="empty",
            quarantine_root=tmp_path,
        )


def test_quarantine_records_child_section_name_and_from_table(
    tmp_path: Path, sample_10k_metadata: ChunkMetadata
) -> None:
    """Audit record carries the new child fields so reviewers can see
    whether the failing child came from a table or narrative."""
    child = _mk_child(sample_10k_metadata, from_table=True, section_name="Item 8")
    cs = ChunkSet(parents=(), children=(child,))
    dest = quarantine_chunkset(
        cs,
        doc_id="test-doc-id",
        source_text="hello",
        reason="x",
        quarantine_root=tmp_path,
    )
    data = json.loads(dest.read_text())
    assert data["children"][0]["from_table"] is True
    assert data["children"][0]["section_name"] == "Item 8"


def test_quarantine_writes_atomically(tmp_path: Path) -> None:
    cs = ChunkSet(parents=(), children=())
    quarantine_chunkset(
        cs,
        doc_id="doc-x",
        source_text="",
        reason="x",
        quarantine_root=tmp_path,
    )
    tmpfiles = list((tmp_path / "chunking").glob(".*tmp*"))
    assert tmpfiles == []


# ---------- validate_or_quarantine_chunkset routing wrapper (ADR / review) --


def test_validate_or_quarantine_returns_chunkset_on_success(
    edge_cases_html: str, edge_metadata: ChunkMetadata, tmp_path: Path
) -> None:
    cs = parse_filing(html=edge_cases_html, metadata=edge_metadata)
    result = validate_or_quarantine_chunkset(
        cs,
        source_text=edge_cases_html,
        doc_id=edge_metadata.doc_id,
        quarantine_root=tmp_path,
    )
    assert result is cs
    # No quarantine file written on success.
    assert not (tmp_path / "chunking").exists()


def test_validate_or_quarantine_returns_none_and_writes_record_on_failure(
    sample_10k_metadata: ChunkMetadata, tmp_path: Path
) -> None:
    # Build a deliberately corrupted ChunkSet whose parent.text doesn't
    # match the source slice.
    bad = _mk_parent(sample_10k_metadata, text="WRONG")
    cs = ChunkSet(parents=(bad,), children=())
    result = validate_or_quarantine_chunkset(
        cs,
        source_text="hello",
        doc_id=sample_10k_metadata.doc_id,
        quarantine_root=tmp_path,
    )
    assert result is None
    dest = tmp_path / "chunking" / f"{sample_10k_metadata.doc_id}.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert "Item 1A" in data["reason"]


# ---------- No-network guarantee --------------------------------------------


def test_parse_filing_makes_no_network_calls(
    edge_cases_html: str, edge_metadata: ChunkMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hermetic guarantee — parsing must not touch the network.

    The session-autouse conftest fixture has already warmed the spaCy
    cache (`_ensure_nlp_warmup`) before this test runs, so the socket
    monkey-patch can't trigger a lazy NLP download.
    """
    import socket

    def _no_socket(*args: Any, **kwargs: Any) -> None:
        raise OSError("network access forbidden during chunking")

    monkeypatch.setattr(socket, "socket", _no_socket)
    monkeypatch.setattr(socket, "create_connection", _no_socket)
    monkeypatch.setattr(socket, "getaddrinfo", _no_socket)

    result = parse_filing(html=edge_cases_html, metadata=edge_metadata)
    assert result.parents


# ---------- Meta-test: no disabling flags on chunking public API ------------


def test_no_disabling_flags_on_chunking_public_api() -> None:
    """INV-2's chunking half has no escape hatch.

    Mirrors `tests/unit/test_extract_guardrails.py::
    test_no_disabling_flags_on_public_api` for the chunking surface.
    A `permissive` / `soft_mode` / `skip_validation` /
    `disable_guardrails` / `lenient` kwarg on any public function in
    `extract.chunking` would be one — fail the suite the moment one
    appears.
    """
    forbidden = {
        "permissive",
        "soft_mode",
        "skip_validation",
        "disable_guardrails",
        "lenient",
    }
    for name in dir(chunking_mod):
        if name.startswith("_"):
            continue
        obj = getattr(chunking_mod, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        offenders = forbidden & set(sig.parameters)
        assert not offenders, f"{name} exposes forbidden kwarg(s): {offenders}"
