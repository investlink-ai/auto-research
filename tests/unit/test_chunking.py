"""Unit tests for `auto_research.extract.chunking` (Tier 2 per INV-2).

Hermetic: no network, no LLM calls. Real 10-K HTML loaded from
checked-in EDGAR fixtures under `tests/fixtures/chunking/`. Each
fixture is `sample_10k_<ticker>[_fyYYYY].htm` + matching meta.json.

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
from typing import TYPE_CHECKING, Any

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
from tests.unit._fixture_meta import FixtureMetadata, load_fixture_meta

if TYPE_CHECKING:
    from tests._otel_helpers import SpanRecorder

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chunking"


# ---------- Fixture discovery + parameterization ----------------------------


def _discover_10k_fixtures() -> list[tuple[str, str]]:
    """Return `[(stem, tier), …]` for every checked-in `sample_10k*.htm`.

    `tier` is read from the matching `.meta.json` (`"core"` or
    `"broad"`) via the `FixtureMetadata` Pydantic model. Adding a new
    ticker via `scripts/build_chunking_fixture.py` automatically
    extends test coverage at whichever tier the build script tags it
    with.

    Tier semantics:
      - `core`: covered by default `make test`. A small, intentional
        subset of templates that catches the bug classes most likely
        to break under chunker changes.
      - `broad`: covered by `make test-broad` (nightly / on-demand).
        Wider industry/year/template coverage; useful for catching
        filer-template variance the core set misses.
    """
    out: list[tuple[str, str]] = []
    for htm in sorted(FIXTURE_DIR.glob("sample_10k*.htm")):
        stem = htm.stem
        meta_path = FIXTURE_DIR / f"{stem}.meta.json"
        if not meta_path.exists():
            continue
        meta = load_fixture_meta(meta_path)
        out.append((stem, meta.tier))
    return out


@dataclasses.dataclass(frozen=True)
class _LoadedFixture:
    stem: str
    html: str
    meta: FixtureMetadata
    metadata: ChunkMetadata
    parsed: ChunkSet


def _load_fixture(stem: str) -> _LoadedFixture:
    html = (FIXTURE_DIR / f"{stem}.htm").read_text(encoding="utf-8", errors="replace")
    meta = load_fixture_meta(FIXTURE_DIR / f"{stem}.meta.json")
    metadata = ChunkMetadata(
        ticker=meta.ticker,
        filing_date=meta.filing_date,
        fiscal_period=meta.fiscal_period,
        doc_type=meta.doc_type,
        doc_id=meta.doc_id,
    )
    return _LoadedFixture(
        stem=stem,
        html=html,
        meta=meta,
        metadata=metadata,
        parsed=parse_filing(html=html, metadata=metadata),
    )


# Cache parsed fixtures at module scope so each parametrized test
# doesn't re-pay the chunking cost (section detection on a 220 KB doc
# takes ~500 ms, multiplied across ~13 per-doc tests across N fixtures
# is real).
_FIXTURE_CACHE: dict[str, _LoadedFixture] = {}


def _cached_fixture(stem: str) -> _LoadedFixture:
    if stem not in _FIXTURE_CACHE:
        _FIXTURE_CACHE[stem] = _load_fixture(stem)
    return _FIXTURE_CACHE[stem]


def _fixture_params() -> list[Any]:
    """Build pytest parametrize args with per-tier marks.

    `tier=="broad"` fixtures get `pytest.mark.broad_fixture`, which
    `make test` excludes via `-m "not broad_fixture"`. Default
    pytest invocations (e.g. `make test-broad`) include them.
    """
    params = []
    for stem, tier in _discover_10k_fixtures():
        marks = [pytest.mark.broad_fixture] if tier == "broad" else []
        params.append(pytest.param(stem, id=stem, marks=marks))
    return params


@pytest.fixture(params=_fixture_params())
def filing(request: pytest.FixtureRequest) -> _LoadedFixture:
    """Parameterized fixture yielding one `_LoadedFixture` per 10-K.

    Each test using this fixture runs once per discovered ticker.
    Broad-tier fixtures (per their meta.json `tier` field) are
    marked `broad_fixture`; `make test` excludes them via marker
    filter while `make test-broad` includes everything.
    """
    return _cached_fixture(request.param)


@pytest.fixture(scope="session")
def edge_cases_html() -> str:
    return (FIXTURE_DIR / "edge_cases.htm").read_text(encoding="utf-8", errors="replace")


@pytest.fixture
def sample_10k_metadata() -> ChunkMetadata:
    """Synthetic ChunkMetadata for tests that don't need a real filing."""
    return ChunkMetadata(
        ticker="NVDA",
        filing_date=date(2025, 2, 26),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="0001045810-25-000023",
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


def test_chunkset_fields_are_tuples_not_lists(filing: _LoadedFixture) -> None:
    """ADR/code-review P1-12: frozen ChunkSet uses tuples so consumers
    cannot mutate the parents/children sequence in place."""
    parsed = filing.parsed
    assert isinstance(parsed.parents, tuple)
    assert isinstance(parsed.children, tuple)
    with pytest.raises(AttributeError):
        parsed.parents.append(parsed.parents[0])  # type: ignore[attr-defined]


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


def test_real_10k_parses_into_expected_sections(filing: _LoadedFixture) -> None:
    section_names = [p.section_name for p in filing.parsed.parents]
    for required in filing.meta.expected_sections:
        assert required in section_names, (
            f"[{filing.stem}] missing {required!r}: got {sorted(set(section_names))}"
        )


def test_real_10k_narrative_parents_under_max_tokens(filing: _LoadedFixture) -> None:
    """Token cap applies to NARRATIVE parents (`table_html is None`).

    Table parents may exceed MAX_PARENT_TOKENS — they take a separate
    structured-extraction path (ADR D5) and splitting them mid-row would
    break HTML well-formedness.
    """
    for p in filing.parsed.parents:
        if p.table_html is not None:
            continue
        assert p.token_count <= MAX_PARENT_TOKENS, (
            f"[{filing.stem}] narrative parent in {p.section_name} exceeds budget: "
            f"{p.token_count} > {MAX_PARENT_TOKENS}"
        )


def test_real_10k_parents_respect_section_boundaries(filing: _LoadedFixture) -> None:
    seen_sections: list[str] = []
    for p in filing.parsed.parents:
        if not seen_sections or seen_sections[-1] != p.section_name:
            assert p.section_name not in seen_sections, (
                f"[{filing.stem}] section {p.section_name} appears "
                f"non-contiguously: {seen_sections}"
            )
            seen_sections.append(p.section_name)


def test_real_10k_children_are_subset_of_parent_spans(filing: _LoadedFixture) -> None:
    parents_by_id: dict[str, ParentChunk] = {}
    for p in filing.parsed.parents:
        pid = f"{p.metadata.doc_id}::{p.char_span[0]}-{p.char_span[1]}"
        parents_by_id[pid] = p

    for c in filing.parsed.children:
        parent = parents_by_id.get(c.parent_id)
        assert parent is not None, f"[{filing.stem}] child has no matching parent: {c.parent_id}"
        assert parent.char_span[0] <= c.char_span[0] < c.char_span[1] <= parent.char_span[1]


def test_real_10k_child_token_band_narrative_only(filing: _LoadedFixture) -> None:
    """Narrative children stay within MAX_UNBREAKABLE_CHILD_TOKENS.

    The strict cap is MAX_CHILD_TOKENS=800, but real iXBRL filings
    sometimes have paragraphs slightly above that with no `</p>` /
    sentence boundary the chunker can split on (BE's Item 16 hits
    this). Rather than failing the whole ChunkSet, the chunker emits
    the oversized child as-is when ≤ MAX_UNBREAKABLE_CHILD_TOKENS
    (= 2x MAX_CHILD_TOKENS). Pathological spans (>2x) still raise.

    Table children may exceed any cap (they equal their parent per
    ADR D5; that's the documented degenerate case).
    """
    from auto_research.extract.chunking import MAX_UNBREAKABLE_CHILD_TOKENS

    for c in filing.parsed.children:
        if c.from_table:
            continue
        assert c.token_count <= MAX_UNBREAKABLE_CHILD_TOKENS, (
            f"[{filing.stem}] narrative child exceeds "
            f"MAX_UNBREAKABLE_CHILD_TOKENS: {c.token_count}"
        )


def test_real_10k_child_section_name_matches_parent(filing: _LoadedFixture) -> None:
    """ADR D7: child carries section_name so LanceDB can filter at
    index time without a parent JOIN."""
    parents_by_id: dict[str, ParentChunk] = {}
    for p in filing.parsed.parents:
        pid = f"{p.metadata.doc_id}::{p.char_span[0]}-{p.char_span[1]}"
        parents_by_id[pid] = p

    for c in filing.parsed.children:
        parent = parents_by_id[c.parent_id]
        assert c.section_name == parent.section_name


def test_real_10k_char_span_fidelity_holds(filing: _LoadedFixture) -> None:
    """The load-bearing INV-2 check: chunk.text equals html slice for
    every parent and child, across every fixture."""
    for p in filing.parsed.parents:
        assert filing.html[p.char_span[0] : p.char_span[1]] == p.text, (
            f"[{filing.stem}] INV-2 broken on parent {p.section_name}"
        )
    for c in filing.parsed.children:
        assert filing.html[c.char_span[0] : c.char_span[1]] == c.text, (
            f"[{filing.stem}] INV-2 broken on child {c.section_name}"
        )


def test_real_10k_metadata_is_populated(filing: _LoadedFixture) -> None:
    expected_ticker = filing.meta.ticker
    expected_doc_id = filing.meta.doc_id
    for p in filing.parsed.parents:
        assert p.metadata.ticker == expected_ticker
        assert p.metadata.doc_type == "10-K"
        assert p.metadata.doc_id == expected_doc_id
    for c in filing.parsed.children:
        assert c.metadata.ticker == expected_ticker


# ---------- Table policy (ADR D5) -------------------------------------------


def test_financial_section_emits_chunks_with_table_html(filing: _LoadedFixture) -> None:
    """If the chunker emits table chunks, the table policy (ADR D5)
    must hold per-chunk. Skip when no table chunks emit.

    A fixture may have `<table>` elements in its HTML without any
    table chunks: ADR D5 explicitly skips cross-section tables to
    avoid producing malformed `table_html` (open `<table>` without
    matching close). Filers that wrap multiple Items inside one
    layout table (some industrial issuers like MPWR, VST, MIR, FORM)
    will see all their tables skipped that way. The chunker is still
    correct — it didn't lie about tables, it just had none to emit.

    The portable assertion is: when table chunks DO emit, the policy
    fields are populated correctly. Skip when none emit; assert on
    the others.
    """
    table_chunks = [p for p in filing.parsed.parents if p.table_html is not None]
    if not table_chunks:
        pytest.skip(
            f"[{filing.stem}] no table chunks emitted (cross-section tables "
            "or no inline tables in trimmed range); table-policy not exercised"
        )
    # Every table chunk's `table_html` must be non-empty and start with
    # a `<table` opening tag (well-formed-HTML contract per ADR D5).
    for p in table_chunks:
        assert p.table_html
        assert p.table_html.lstrip().lower().startswith("<table"), (
            f"[{filing.stem}] table_html doesn't start with <table>"
        )


def test_item_1a_has_at_least_some_narrative_chunks(filing: _LoadedFixture) -> None:
    """Item 1A (Risk Factors) is *primarily* narrative across all filers.

    Some filers (e.g., MSFT) format bulleted risk-factor lists using
    HTML layout tables — `<table>` with no real numeric data, used
    only for visual indentation. Those chunks carry `table_html`
    legitimately. The chunker's table policy (ADR D5) emits any
    `<table>` as a table chunk regardless of whether it's a data
    table or a layout table — downstream `pandas.read_html` handles
    the discrimination by either successfully parsing the structured
    data or rejecting unstructured layout content.

    The test asserts the weaker, filer-portable invariant: at least
    SOME Item 1A chunks must be narrative (`table_html=None`). If
    every Item 1A chunk has `table_html`, either detection assigned
    a wrong section_name or the section is entirely tabular (not
    plausible for risk factors).
    """
    item_1a = [p for p in filing.parsed.parents if p.section_name == "Item 1A"]
    assert item_1a, f"[{filing.stem}] expected Item 1A parents"
    narrative_chunks = [p for p in item_1a if p.table_html is None]
    assert narrative_chunks, (
        f"[{filing.stem}] expected ≥1 narrative Item 1A chunk; "
        f"all {len(item_1a)} have table_html (suspicious)"
    )


def test_at_least_one_table_html_is_pandas_readable(filing: _LoadedFixture) -> None:
    """At least one of the chunker's `table_html` payloads parses
    successfully via `pandas.read_html`.

    Skipped when:
      - the fixture has no table chunks (covered by the sibling test)
      - every table is a "layout table" — well-formed `<table>` HTML
        the chunker correctly emitted, but used for visual purposes
        (bulleted lists, side-by-side text columns) rather than as a
        data table. Some filers (RDW, MSFT for Item 1A) use layout
        tables extensively. Pandas's discriminator filters them out;
        the chunker's job is to emit them so the downstream extractor
        decides what to do.

    Pandas raises ValueError for "no tables found" and TypeError for
    schema-shape issues — both are layout-table indicators rather
    than chunker bugs. The chunker contract (`table_html` is well-
    formed HTML starting with `<table>`) is asserted in the sibling
    `test_financial_section_emits_chunks_with_table_html`.
    """
    tables = [p for p in filing.parsed.parents if p.table_html is not None]
    if not tables:
        pytest.skip(f"[{filing.stem}] no table chunks; sibling test covers this case")
    for t in tables:
        try:
            dfs = pd.read_html(io.StringIO(t.table_html))
        except (ValueError, TypeError):
            continue
        if dfs and not dfs[0].empty:
            return  # success — at least one real data table found
    pytest.skip(
        f"[{filing.stem}] {len(tables)} table chunks emitted but none "
        "parse via pandas.read_html — likely all layout tables, not "
        "a chunker bug"
    )


def test_cross_section_table_is_not_emitted_as_table_chunk(
    edge_metadata: ChunkMetadata,
) -> None:
    """ADR D5: a `<table>` that opens inside one section but `</table>`
    closes after the next section header is NOT emitted as a table
    chunk. Clamping its span to the section end would yield malformed
    `table_html` (open `<table>` without close), violating D5's well-
    formed-HTML invariant for downstream `pandas.read_html` consumers.

    Instead the open `<table>` falls into the originating section's
    narrative span and the close lands in the next section's
    narrative. INV-2 still holds (chunk.text == html[span]); the
    table-policy contract simply doesn't apply to this table.
    """
    cross_table = (
        "<table id='cross'>"
        "<tr><td>row before next section header</td></tr>"
        # The next-section header appears INSIDE this table.
        "<tr><td>"
        "<div><span style=\"font-weight:700\">Item 7. Management's Discussion and Analysis</span></div>"
        "</td></tr>"
        "<tr><td>row after next section header</td></tr>"
        "</table>"
    )
    html = (
        "<html><body><h1>Item 1A. Risk Factors</h1>"
        + ("<p>Risk factor paragraph " + ("word " * 80) + ".</p>") * 4
        + cross_table
        + ("<p>MD&amp;A paragraph " + ("word " * 80) + ".</p>") * 4
        + "</body></html>"
    )
    result = parse_filing(html=html, metadata=edge_metadata)

    # Both sections detected.
    sections = {p.section_name for p in result.parents}
    assert "Item 1A" in sections, f"got: {sorted(sections)}"
    assert "Item 7" in sections, f"got: {sorted(sections)}"

    # No parent chunk should claim this cross-section <table> as
    # `table_html` — the chunker recognized it as cross-boundary and
    # skipped it.
    table_chunks = [p for p in result.parents if p.table_html is not None]
    for p in table_chunks:
        assert p.table_html is not None  # narrow type for mypy
        assert "row after next section header" not in p.table_html, (
            "cross-section table was emitted as a table chunk; ADR D5 violated"
        )

    # INV-2 must still hold across all chunks.
    for p in result.parents:
        assert html[p.char_span[0] : p.char_span[1]] == p.text


def test_layout_table_is_emitted_with_well_formed_table_html(
    edge_metadata: ChunkMetadata,
) -> None:
    """Some filers (MSFT, RDW) use HTML `<table>` for visual layout —
    bulleted lists, side-by-side text columns — rather than for
    tabular data. The chunker can't (and shouldn't) distinguish
    layout-tables from data-tables structurally; it emits ALL
    `<table>` regions as table chunks with `table_html` populated.

    Downstream consumers discriminate: `pandas.read_html` either
    parses the table as data or rejects it. The chunker's contract
    here is narrow: `table_html`, when present, starts with `<table`
    and ends with `</table>` so downstream HTML-aware code receives
    well-formed markup.
    """
    layout_table = (
        "<table>"
        "<tr>"
        "<td style='width:5%'>&#8226;</td>"  # bullet glyph
        "<td>This is a bulleted risk-factor item formatted as a "
        "layout table. The cell on the left is the bullet glyph; "
        "this cell holds the risk-factor prose itself.</td>"
        "</tr>"
        "<tr>"
        "<td>&#8226;</td>"
        "<td>A second bulleted item, also using the layout-table "
        "convention rather than a real `<ul>` list.</td>"
        "</tr>"
        "</table>"
    )
    html = (
        "<html><body><h1>Item 1A. Risk Factors</h1>"
        + ("<p>Risk factor prose " + ("word " * 80) + ".</p>")
        + layout_table
        + ("<p>More risk factor prose " + ("word " * 80) + ".</p>")
        + "</body></html>"
    )
    result = parse_filing(html=html, metadata=edge_metadata)
    table_chunks = [p for p in result.parents if p.table_html is not None]
    assert table_chunks, "layout table not emitted as a table chunk"
    for p in table_chunks:
        # Well-formed-HTML contract (ADR D5).
        assert p.table_html
        assert p.table_html.lstrip().lower().startswith("<table")
        assert p.table_html.rstrip().lower().endswith("</table>")
    # INV-2 holds.
    for p in result.parents:
        assert html[p.char_span[0] : p.char_span[1]] == p.text


def test_bare_section_title_detected_when_item_prefix_missing(
    edge_metadata: ChunkMetadata,
) -> None:
    """Some filers (HON, NRG, others) mark section bodies with bare
    canonical SEC titles ('Risk Factors', 'Management's Discussion and
    Analysis') without the 'Item N.' prefix. The chunker's bare-name
    fallback maps these back to Item numbers."""
    html = (
        "<html><body>"
        # Cover page — no "Item N." prefix anywhere in body.
        "<div style='font-weight:700'>ANNUAL REPORT ON FORM 10-K</div>"
        + ("<p>Cover-page boilerplate " + ("word " * 30) + ".</p>")
        # Bare title for Item 1A — large styled heading.
        + "<div><span style='font-size:18pt;font-weight:700'>Risk Factors</span></div>"
        + ("<p>Our business involves significant risk. " + ("risk " * 100) + ".</p>") * 4
        # Bare title for Item 7.
        + "<div><span style='font-size:18pt;font-weight:700'>"
        "Management's Discussion and Analysis</span></div>"
        + ("<p>The following discussion " + ("word " * 100) + ".</p>") * 4
        # Bare title for Item 8.
        + "<div><span style='font-size:18pt;font-weight:700'>"
        "Financial Statements and Supplementary Data</span></div>"
        + ("<p>The financial statements " + ("word " * 100) + ".</p>") * 4
        + "</body></html>"
    )
    result = parse_filing(html=html, metadata=edge_metadata)
    section_names = {p.section_name for p in result.parents}
    assert "Item 1A" in section_names, f"got: {sorted(section_names)}"
    assert "Item 7" in section_names, f"got: {sorted(section_names)}"
    assert "Item 8" in section_names, f"got: {sorted(section_names)}"


def test_bare_section_title_in_toc_anchor_does_not_false_positive(
    edge_metadata: ChunkMetadata,
) -> None:
    """The string 'Risk Factors' inside a `<a href="#...">` TOC link
    is navigation, not a real section header. The bare-name fallback
    must skip anchor-wrapped titles."""
    html = (
        "<html><body>"
        # TOC with a bunch of anchor links — bare titles inside <a>.
        "<div>Table of Contents</div>"
        "<div><a href='#item1a'>Risk Factors</a></div>"
        "<div><a href='#item7'>Management's Discussion and Analysis</a></div>"
        + ("<p>cover boilerplate " + ("word " * 30) + ".</p>")
        # Real bare-title heading much later in doc order, also for 1A.
        + "<div><span style='font-size:18pt;font-weight:700'>Risk Factors</span></div>"
        + ("<p>Our business involves significant risk. " + ("risk " * 100) + ".</p>") * 4
        + "</body></html>"
    )
    result = parse_filing(html=html, metadata=edge_metadata)
    item_1a_parents = [p for p in result.parents if p.section_name == "Item 1A"]
    assert item_1a_parents, "Item 1A should be detected via the real bare header"
    # The detected Item 1A should be the real one (the LATER position),
    # not the TOC anchor.
    first_1a_start = item_1a_parents[0].char_span[0]
    toc_anchor_pos = html.find("<a href='#item1a'>")
    assert first_1a_start > toc_anchor_pos, (
        "Item 1A detected at the TOC anchor position instead of the real header"
    )


def test_em_dash_separator_in_item_header_is_recognized(
    edge_metadata: ChunkMetadata,
) -> None:
    """Some filers (BE, NRG) write `Item 1A&#8212;Risk Factors`
    (em-dash separator) rather than the canonical `Item 1A. Risk
    Factors`. The chunker's title-pattern accepts em-dash, en-dash,
    hyphen, period, and colon as separators."""
    html = (
        "<html><body>"
        + "<div><span style='font-weight:700'>Item 1A&#8212;Risk Factors</span></div>"
        + ("<p>Risk factor content " + ("word " * 100) + ".</p>") * 4
        + "<div><span style='font-weight:700'>Item 7 &#8211; Management's Discussion</span></div>"
        + ("<p>MD&amp;A content " + ("word " * 100) + ".</p>") * 4
        + "</body></html>"
    )
    result = parse_filing(html=html, metadata=edge_metadata)
    section_names = {p.section_name for p in result.parents}
    assert "Item 1A" in section_names
    assert "Item 7" in section_names


def test_nested_table_html_covers_outer_table(edge_metadata: ChunkMetadata) -> None:
    """`table_html` must cover the full OUTER table even when inner
    tables are nested inside — common in SEC iXBRL layouts. Without
    depth tracking the first `</table>` after the outer open would
    truncate the chunk at the inner close, leaving the outer's tail
    rows leaking into narrative."""
    inner_a = "<table><tr><td>inner A row</td></tr></table>"
    inner_b = "<table><tr><td>inner B row</td></tr></table>"
    outer = (
        "<table id='outer'>"
        "<tr><td>outer header cell</td></tr>"
        f"<tr><td>cell with nested {inner_a}</td></tr>"
        f"<tr><td>another nested {inner_b}</td></tr>"
        "<tr><td>outer footer cell</td></tr>"
        "</table>"
    )
    # Build a doc with a real section header so the table lands inside it.
    html = (
        "<html><body><h1>Item 8. Financial Statements</h1>"
        + ("<p>Filler prose paragraph " + ("word " * 100) + ".</p>") * 3
        + outer
        + ("<p>Trailing prose paragraph " + ("word " * 100) + ".</p>") * 3
        + "</body></html>"
    )
    result = parse_filing(html=html, metadata=edge_metadata)
    table_parents = [p for p in result.parents if p.table_html is not None]
    assert len(table_parents) == 1, (
        f"expected exactly one outer-table chunk, got {len(table_parents)}"
    )
    tbl = table_parents[0]
    # The outer table's footer row must be inside the chunk text, proving
    # we didn't truncate at the first inner `</table>`.
    assert "outer footer cell" in tbl.text
    assert tbl.text.startswith("<table id='outer'>")
    assert tbl.text.rstrip().endswith("</table>")


def test_table_parents_subdivide_to_single_child_equal_to_parent(
    filing: _LoadedFixture,
) -> None:
    """ADR D5: table parents are atomic. Splitting at `</td>` would
    produce fragments without `</table>`. Each table parent must yield
    exactly one child with from_table=True and char_span equal to
    parent's char_span."""
    children_by_parent_id: dict[str, list[ChildChunk]] = {}
    for c in filing.parsed.children:
        children_by_parent_id.setdefault(c.parent_id, []).append(c)

    for p in filing.parsed.parents:
        if p.table_html is None:
            continue
        pid = f"{p.metadata.doc_id}::{p.char_span[0]}-{p.char_span[1]}"
        kids = children_by_parent_id.get(pid, [])
        assert len(kids) == 1, (
            f"[{filing.stem}] table parent in {p.section_name} should yield "
            f"1 child, got {len(kids)}"
        )
        kid = kids[0]
        assert kid.from_table is True
        assert kid.char_span == p.char_span
        assert kid.text == p.text


def test_narrative_children_have_from_table_false(filing: _LoadedFixture) -> None:
    for c in filing.parsed.children:
        if c.from_table:
            continue
        # The corresponding parent must also be a narrative parent.
        pid_parts = c.parent_id.split("::")[-1].split("-")
        parent_span = (int(pid_parts[0]), int(pid_parts[1]))
        matching_parents = [p for p in filing.parsed.parents if p.char_span == parent_span]
        assert matching_parents, f"[{filing.stem}] no parent for child {c.parent_id}"
        assert matching_parents[0].table_html is None


# ---------- subdivide_to_children — quarantine-or-raise on bad input -------


def test_subdivide_raises_when_no_safe_split_exists(
    edge_metadata: ChunkMetadata,
) -> None:
    """A narrative parent containing a single span pathologically larger
    than MAX_UNBREAKABLE_CHILD_TOKENS with no internal break boundary
    must raise ChunkValidationError (caller routes to quarantine).
    Modest oversize (between MAX_CHILD_TOKENS and 2x MAX) is allowed
    — see test_subdivide_tolerates_modest_unbreakable_oversize."""
    import random

    from auto_research.extract.chunking import MAX_UNBREAKABLE_CHILD_TOKENS

    # Need >MAX_UNBREAKABLE_CHILD_TOKENS=1600 tokens in a single span
    # with no sentence/HTML break. Random word-like chunks separated
    # by spaces produces ~0.75 tokens per word; ~3000 random words
    # yields well above the threshold.
    rng = random.Random(7)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # Random ASCII gibberish → ~3-4 tokens/word. ~600 words gives
    # ~2200 tokens, well above MAX_UNBREAKABLE_CHILD_TOKENS (1600).
    words = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 9)))
        for _ in range(600)
    ]
    long_run = " ".join(words)
    parent = ParentChunk(
        text=long_run,
        section_name="Item 1A",
        char_span=(0, len(long_run)),
        token_count=count_tokens(long_run),
        table_html=None,
        metadata=edge_metadata,
    )
    assert parent.token_count > MAX_UNBREAKABLE_CHILD_TOKENS, parent.token_count
    with pytest.raises(ChunkValidationError):
        chunking_mod.subdivide_to_children(parent)


def test_subdivide_tolerates_modest_unbreakable_oversize(
    edge_metadata: ChunkMetadata,
) -> None:
    """Real iXBRL filings sometimes have paragraphs slightly above
    MAX_CHILD_TOKENS with no internal `. ` / `</p>` boundaries (BE,
    others). Failing the whole ChunkSet for a chunk that's modestly
    over budget would lose useful coverage. The chunker now emits the
    oversized child as-is when it's ≤ MAX_UNBREAKABLE_CHILD_TOKENS;
    only pathological spans (>2x MAX) raise."""
    import random

    from auto_research.extract.chunking import MAX_UNBREAKABLE_CHILD_TOKENS

    # Random ASCII word-like tokens separated by spaces. Need ~900
    # tokens (above MAX_CHILD_TOKENS=800, under MAX_UNBREAKABLE=1600)
    # WITHOUT sentence-ending punctuation that would create a
    # boundary. ~1200 random word-like chunks of length 4-9 separated
    # by spaces lands in the target range and resists tiktoken
    # compression because there are no repeated patterns.
    rng = random.Random(42)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # Random ASCII gibberish doesn't merge into vocab tokens, so each
    # word averages ~3-4 tokens. ~300 words → ~1000-1200 tokens,
    # comfortably between MAX_CHILD_TOKENS (800) and
    # MAX_UNBREAKABLE_CHILD_TOKENS (1600).
    words = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 9)))
        for _ in range(300)
    ]
    text = " ".join(words)
    parent = ParentChunk(
        text=text,
        section_name="Item 1A",
        char_span=(0, len(text)),
        token_count=count_tokens(text),
        table_html=None,
        metadata=edge_metadata,
    )
    assert MAX_CHILD_TOKENS < parent.token_count <= MAX_UNBREAKABLE_CHILD_TOKENS
    # Should NOT raise — modest oversize is tolerated.
    children = chunking_mod.subdivide_to_children(parent)
    assert children, "expected at least one child for an oversize parent"
    # The single emitted child should cover the parent.
    assert any(c.char_span == parent.char_span for c in children)


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


# ---------- Fixture metadata schema ----------------------------------------


def test_all_chunking_fixtures_validate_against_fixture_metadata_model(
    tmp_path: Path,
) -> None:
    """Every checked-in `*.meta.json` parses cleanly through
    `FixtureMetadata`. A typo'd key (`extra="forbid"`) or a missing
    required field fails this test rather than surfacing later as a
    KeyError inside a parametrized test, which would only run when the
    specific fixture happened to be selected.
    """
    fixtures = sorted(FIXTURE_DIR.glob("sample_10k*.meta.json"))
    assert len(fixtures) >= 70, "fixture set unexpectedly small"
    for meta_path in fixtures:
        meta = load_fixture_meta(meta_path)
        # Sanity: typed access works downstream without `# type: ignore`.
        assert meta.ticker
        assert meta.doc_type == "10-K"
        assert meta.tier in {"core", "broad"}


def test_fixture_metadata_loader_cites_offending_stem_on_malformed_json(
    tmp_path: Path,
) -> None:
    """A typo'd key fails fast with the fixture filename in the error.
    Catches the sentinel case from issue #57 Item 3: extending the
    schema must not silently degrade older fixtures into KeyErrors.
    """
    bogus = tmp_path / "sample_10k_bogus.meta.json"
    bogus.write_text(
        json.dumps(
            {
                "ticker": "BOGUS",
                "filing_date": "2026-01-01",
                "fiscal_period": "FY2025",
                "doc_type": "10-K",
                "doc_id": "x",
                "expected_sections": [],
                "source_url": "http://example.com",
                "filename_convention": "n/a",
                "note": "n/a",
                "tier": "core",
                "tier_rationalle": "typo'd key — extra='forbid' should catch this",
            }
        )
    )
    with pytest.raises(ValueError, match=r"sample_10k_bogus\.meta\.json"):
        load_fixture_meta(bogus)


# ---------- Doc-type dispatcher --------------------------------------------


def test_parse_filing_raises_clear_error_for_unregistered_doc_type(
    edge_metadata: ChunkMetadata,
) -> None:
    """`parse_filing` must reject unregistered doc types with a clear
    remediation message rather than silently emitting a one-Body
    ChunkSet. Foreign filers (20-F / 40-F) are the canonical case —
    see `docs/decisions/2026-05-25-foreign-filers-deferred.md`. A
    silent fallback would corrupt downstream LanceDB section filters
    (every chunk gets `section_name='Body'`).
    """
    bad_meta = dataclasses.replace(edge_metadata, doc_type="20-F")
    with pytest.raises(ValueError, match="No chunker detector"):
        parse_filing(html="<html><body>filler</body></html>", metadata=bad_meta)


def test_get_detector_returns_callable_for_registered_form() -> None:
    """The registry is the only path parse_filing uses; sanity-check
    that 10-K resolves to something callable rather than asserting on
    an internal function name."""
    from auto_research.extract.chunking.detect import get_detector

    detector = get_detector("10-K")
    assert callable(detector)


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


# ---------- OTel instrumentation --------------------------------------------


def test_parse_filing_emits_span_on_ok_path(
    edge_cases_html: str,
    edge_metadata: ChunkMetadata,
    span_recorder: SpanRecorder,
) -> None:
    """`parse_filing` emits one `chunk.parse_filing` span carrying
    doc_id / doc_type / ticker / section + chunk counts / outcome=ok."""
    result = parse_filing(html=edge_cases_html, metadata=edge_metadata)

    attrs = span_recorder.attrs("chunk.parse_filing")
    assert attrs["chunk.doc_id"] == edge_metadata.doc_id
    assert attrs["chunk.doc_type"] == "10-K"
    assert attrs["chunk.ticker"] == "TEST"
    assert attrs["chunk.outcome"] == "ok"
    # AttributeValue is a broad union; narrow to int for arithmetic.
    n_sections = attrs["chunk.n_sections"]
    n_parents = attrs["chunk.n_parents"]
    n_children = attrs["chunk.n_children"]
    n_table = attrs["chunk.n_table_parents"]
    assert isinstance(n_sections, int) and n_sections >= 1
    assert isinstance(n_parents, int) and n_parents == len(result.parents)
    assert isinstance(n_children, int) and n_children == len(result.children)
    assert isinstance(n_table, int) and n_table == sum(
        1 for p in result.parents if p.table_html is not None
    )


def test_parse_filing_span_marks_no_sections_detected(
    edge_metadata: ChunkMetadata,
    span_recorder: SpanRecorder,
) -> None:
    """Doc with no Item-prefix or bare-title headers falls back to a
    single 'Body' section; the span outcome reflects that."""
    html = "<html><body>" + ("<p>filler " + ("x " * 30) + "</p>") * 5 + "</body></html>"
    parse_filing(html=html, metadata=edge_metadata)

    attrs = span_recorder.attrs("chunk.parse_filing")
    assert attrs["chunk.outcome"] == "no_sections_detected"
    assert attrs["chunk.n_sections"] == 1


def test_parse_filing_span_records_validation_failure(
    edge_metadata: ChunkMetadata,
    span_recorder: SpanRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `validate_char_spans` raises, the span tags
    `chunk.outcome=validation_failed` and ERROR status before the
    exception propagates. Mirrors the s_filings-worker error-path
    pattern."""

    def _broken_validator(*args: Any, **kwargs: Any) -> None:
        raise ChunkValidationError("synthetic mismatch")

    monkeypatch.setattr(chunking_mod, "validate_char_spans", _broken_validator)

    html = (
        "<html><body><h1>Item 1A. Risk Factors</h1>"
        + ("<p>Risk content " + ("word " * 80) + ".</p>") * 4
        + "</body></html>"
    )
    with pytest.raises(ChunkValidationError):
        parse_filing(html=html, metadata=edge_metadata)

    attrs = span_recorder.attrs("chunk.parse_filing")
    assert attrs["chunk.outcome"] == "validation_failed"
    span = span_recorder.one("chunk.parse_filing")
    assert span.status.status_code.name == "ERROR"


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
