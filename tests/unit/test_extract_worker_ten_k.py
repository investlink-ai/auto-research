"""Unit tests for the 10-K extraction worker.

Hermetic — the Anthropic SDK is mocked. Three branch-coverage tests
satisfy AC bullet 3 ("Hybrid extraction policy: single-shot for
< SINGLE_SHOT_TOKEN_CUTOFF tokens, RAG path for ≥") plus AC bullet 4
("10-K Item 8 financials extracted from ParentChunk.table_html via
typed Pydantic schema").

The RAG path uses an injected `retrieve_fn` so the test doesn't have
to stand up the full hybrid_retrieve + rerank stack.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from auto_research.extract.chunking import (
    ChildChunk,
    ChunkMetadata,
    ChunkSet,
    ParentChunk,
)
from auto_research.extract.schemas import (
    Citation,
    FinancialLineItem,
    TenKFinancials,
)
from auto_research.extract.workers.ten_k import (
    _merge_financials,
    _render_table_html_to_text,
    extract_ten_k,
)
from tests.unit.conftest import (
    make_fake_anthropic_client as _fake_client_single,
)
from tests.unit.conftest import (
    make_fake_anthropic_client_sequence as _fake_client_sequence,
)

_SAMPLE_10K = (
    "Item 1A. Risk Factors. Our supply chain depends on TSMC.\n"
    "Item 7. Management's Discussion and Analysis. "
    "We expect cautious growth in fiscal 2026.\n"
)


def _valid_narrative() -> dict[str, Any]:
    return {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
        "guidance_tone": {
            "citation": {"source_quote": "cautious growth in fiscal 2026"},
            "confidence": "high",
        },
        "accrual_flags": [],
        "supplier_mentions": [
            {
                "mention_text": "TSMC",
                "citation": {"source_quote": "TSMC"},
                "resolved_ticker": None,
                "resolver_confidence": None,
                "resolver_reasoning": None,
            }
        ],
        "customer_mentions": [],
        "language_novelty_score": 0.0,
        "risk_factor_deltas": [],
    }


def _valid_financials() -> dict[str, Any]:
    return {
        "revenue": {
            "value_usd": 1234567.0,
            "citation": {"source_quote": "Revenue"},
            "confidence": "high",
        },
        "gross_profit": None,
        "operating_income": None,
        "net_income": {
            "value_usd": 456789.0,
            "citation": {"source_quote": "Net income"},
            "confidence": "high",
        },
        "total_assets": None,
        "total_liabilities": None,
        "stockholders_equity": None,
        "cash_from_operations": None,
        "cash_from_investing": None,
        "cash_from_financing": None,
    }


def _chunkset_with_table(table_html: str) -> ChunkSet:
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-table-001",
    )
    parent = ParentChunk(
        text=table_html,
        section_name="item_8",
        char_span=(0, len(table_html)),
        token_count=10,
        table_html=table_html,
        metadata=meta,
    )
    child = ChildChunk(
        text=table_html,
        char_span=(0, len(table_html)),
        token_count=10,
        parent_id="x",
        section_name="item_8",
        from_table=True,
        metadata=meta,
    )
    return ChunkSet(parents=(parent,), children=(child,))


def _chunkset_narrative_only() -> ChunkSet:
    """A chunkset whose parents have NO table_html — exercises the
    'RAG narrative branch, no Item 8 financials' code path."""
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-rag-001",
    )
    parent_text = (
        "Item 7. Management's Discussion and Analysis. "
        "We expect cautious growth in fiscal 2026."
    )
    parent = ParentChunk(
        text=parent_text,
        section_name="item_7",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    return ChunkSet(parents=(parent,), children=())


# --- Branch coverage: single-shot --------------------------------------------


def test_ten_k_single_shot_branch_no_chunkset(tmp_path: Path) -> None:
    """Short raw doc, no chunkset → single-shot. Exactly one LLM call;
    financials remains None."""
    client = _fake_client_single(_valid_narrative())
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.fiscal_period_end == date(2025, 12, 31)
    assert out.financials is None
    assert out.supplier_mentions[0].mention_text == "TSMC"
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]


def test_ten_k_single_shot_branch_short_doc_with_narrative_chunkset(
    tmp_path: Path,
) -> None:
    """Short raw doc + chunkset (no table parents) → still single-shot
    narrative (chunkset alone does NOT trigger the RAG path; the size
    threshold does). Item 8 doesn't fire because there's no table parent."""
    client = _fake_client_single(_valid_narrative())
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-002",
        cache_root=tmp_path,
        anthropic_client=client,
        chunkset=_chunkset_narrative_only(),
    )
    assert out is not None
    assert out.financials is None
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]


# --- Branch coverage: RAG ---------------------------------------------------


def test_ten_k_rag_branch_fires_above_cutoff(tmp_path: Path) -> None:
    """count_tokens(raw_doc) >= SINGLE_SHOT_TOKEN_CUTOFF AND chunkset
    supplied → RAG branch. One LLM call per narrative field (5), each
    against distinct user_content (per-field retrieve), producing 5
    distinct cache keys and 5 LLM calls.

    Each per-field parent text contains a unique sentinel quote (the
    field's name appended to "FIELD-"). Each response cites that
    sentinel for its guidance_tone Claim (a required schema field that
    must resolve) — this keeps every call's schema validation happy
    while keeping user_content + cache key distinct per call.
    """
    long_raw = "word " * 200_000  # ~200K tokens — well above cutoff
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-rag-001",
    )
    field_to_keyword = {
        "guidance_tone": "growth",
        "accrual_flags": "accrual",
        "supplier_mentions": "supplier",
        "customer_mentions": "customer",
        "risk_factor_deltas": "risk",
    }
    # Each parent's text contains a unique sentinel "FIELD-<field>" that
    # serves as the response's source_quote. Distinct text → distinct
    # cache key per call.
    per_field_parents: dict[str, list[ParentChunk]] = {
        field: [
            ParentChunk(
                text=f"FIELD-{field} marker in this parent passage",
                section_name="item_7",
                char_span=(0, 1),
                token_count=10,
                table_html=None,
                metadata=meta,
            )
        ]
        for field in field_to_keyword
    }

    queries_seen: list[str] = []

    def fake_retrieve(query: str) -> list[ParentChunk]:
        queries_seen.append(query)
        for field, keyword in field_to_keyword.items():
            if keyword in query.lower():
                return per_field_parents[field]
        return []

    def _response_for(field: str) -> dict[str, Any]:
        sentinel = f"FIELD-{field}"
        base = {
            "cik": "0000000001",
            "accession_number": "0000000001-26-000001",
            "fiscal_period_end": "2025-12-31",
        }
        if field == "guidance_tone":
            return {
                **base,
                "guidance_tone": {
                    "citation": {"source_quote": sentinel},
                    "confidence": "high",
                },
            }
        if field == "accrual_flags":
            return {
                **base,
                "accrual_flags": [
                    {
                        "citation": {"source_quote": sentinel},
                        "confidence": "medium",
                    }
                ],
            }
        if field == "supplier_mentions":
            return {
                **base,
                "supplier_mentions": [
                    {
                        "mention_text": "Sentinel Supplier Inc.",
                        "citation": {"source_quote": sentinel},
                        "resolved_ticker": None,
                        "resolver_confidence": None,
                        "resolver_reasoning": None,
                    }
                ],
            }
        if field == "customer_mentions":
            return {
                **base,
                "customer_mentions": [
                    {
                        "mention_text": "Sentinel Customer Corp.",
                        "citation": {"source_quote": sentinel},
                        "resolved_ticker": None,
                        "resolver_confidence": None,
                        "resolver_reasoning": None,
                    }
                ],
            }
        if field == "risk_factor_deltas":
            return {
                **base,
                "risk_factor_deltas": [
                    {
                        "change_type": "added",
                        "text": "new risk factor language",
                        "citation": {"source_quote": sentinel},
                    }
                ],
            }
        raise ValueError(f"unknown field {field!r}")

    # The worker iterates `_NARRATIVE_RAG_QUERIES` in dict order, which
    # is insertion order: guidance_tone, accrual_flags, supplier_mentions,
    # customer_mentions, risk_factor_deltas.
    responses_in_order = [
        _response_for(f)
        for f in (
            "guidance_tone",
            "accrual_flags",
            "supplier_mentions",
            "customer_mentions",
            "risk_factor_deltas",
        )
    ]
    client = _fake_client_sequence(responses_in_order)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-rag-001",
        cache_root=tmp_path,
        anthropic_client=client,
        chunkset=_chunkset_narrative_only(),
        retrieve_fn=fake_retrieve,
    )
    assert out is not None
    assert len(queries_seen) == 5  # one query per narrative field
    assert client.messages.create.call_count == 5  # type: ignore[attr-defined]
    # No table parents → Item 8 doesn't fire.
    assert out.financials is None


def test_ten_k_rag_branch_requires_retrieve_fn(tmp_path: Path) -> None:
    """Above the cutoff with chunkset but no retrieve_fn → ValueError,
    not a silent fallback to single-shot."""
    long_raw = "word " * 200_000
    chunkset = _chunkset_narrative_only()
    client = _fake_client_single(_valid_narrative())
    with pytest.raises(ValueError, match="retrieve_fn"):
        extract_ten_k(
            raw_doc=long_raw,
            doc_id="10k-rag-fail",
            cache_root=tmp_path,
            anthropic_client=client,
            chunkset=chunkset,
        )


def _rag_partials_in_order(cik: str = "0000000001") -> list[dict[str, Any]]:
    """Five partial-schema dicts (one per narrative field) whose
    citation quote ('cautious growth in fiscal 2026') is present in
    `_chunkset_narrative_only`'s parent text — so each per-field
    response validates cleanly against its respective Pydantic partial.

    Order matches `TEN_K_NARRATIVE_FIELD_CONFIGS`: guidance_tone,
    accrual_flags, supplier_mentions, customer_mentions,
    risk_factor_deltas.
    """
    base = {
        "cik": cik,
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    quote = "cautious growth in fiscal 2026"
    return [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": quote},
                "confidence": "high",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
    ]


def test_ten_k_rag_partial_failure_does_not_persist_earlier_fields(
    tmp_path: Path,
) -> None:
    """If field 3 quarantines, fields 1+2's outputs MUST NOT be in the
    cache — otherwise a re-run hits stale entries from this attempt
    while re-running the failed fields, producing non-deterministic
    partial state."""
    long_raw = "word " * 200_000
    chunkset = _chunkset_narrative_only()

    valid_in_order = _rag_partials_in_order()
    bad_supplier = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
        "supplier_mentions": [
            {
                "mention_text": "Phantom Co.",
                "citation": {"source_quote": "not in any parent"},
                "resolved_ticker": None,
                "resolver_confidence": None,
                "resolver_reasoning": None,
            }
        ],
    }
    client = _fake_client_sequence(
        [
            valid_in_order[0],  # guidance_tone — succeeds, stages
            valid_in_order[1],  # accrual_flags — succeeds, stages
            bad_supplier,  # supplier_mentions — fails, quarantines
        ]
    )
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-rag-partial",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: list(chunkset.parents),
    )
    assert out is None
    # No cache files written — staged writes are dropped on failure.
    cache_files = list((tmp_path / "cache").rglob("*.json"))
    assert cache_files == [], f"unexpected cache state: {cache_files}"


def test_ten_k_rag_identity_disagreement_quarantines(tmp_path: Path) -> None:
    """If 5 per-field RAG calls disagree on cik / accession_number /
    fiscal_period_end, the worker MUST quarantine rather than silently
    keep the first call's value."""
    long_raw = "word " * 200_000
    chunkset = _chunkset_narrative_only()
    base = _rag_partials_in_order(cik="0000000001")
    diverged = _rag_partials_in_order(cik="0000999999")
    # Swap the second call (accrual_flags) for a partial with a
    # diverged cik. The remaining 4 keep the agreed cik.
    client = _fake_client_sequence(
        [base[0], diverged[1], base[2], base[3], base[4]]
    )
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-rag-identity",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: list(chunkset.parents),
    )
    assert out is None
    qrec = tmp_path / "q" / "ten_k" / "10k-rag-identity#identity-disagreement.json"
    assert qrec.exists()
    record = json.loads(qrec.read_text())
    assert "disagree" in record["error"]
    assert "cik" in record["error"]
    # Identity check fires BEFORE commit, so staged writes are dropped
    # and the cache stays empty even though all 5 calls validated.
    assert list((tmp_path / "cache").rglob("*.json")) == []


def test_ten_k_long_doc_without_chunkset_raises(tmp_path: Path) -> None:
    """A long raw_doc with no chunkset MUST raise, not silently send the
    full doc as one user_content block. Silent fallthrough would bill the
    full 100K+ tokens as fresh input every call and likely exceed the
    model's input window."""
    long_raw = "word " * 200_000  # ~200K tokens, well above SINGLE_SHOT cutoff
    client = _fake_client_single(_valid_narrative())
    with pytest.raises(ValueError, match="cutoff"):
        extract_ten_k(
            raw_doc=long_raw,
            doc_id="10k-long-no-chunkset",
            cache_root=tmp_path,
            anthropic_client=client,
        )


# --- Item 8 path ------------------------------------------------------------


def test_ten_k_item8_financials_extracted_from_table_html(tmp_path: Path) -> None:
    """Chunkset with a table parent → two LLM calls: narrative + Item 8
    financials. The resulting TenKOutput carries the financials data."""
    table_html = (
        "<table>"
        "<tr><td>Revenue</td><td>$1,234,567</td></tr>"
        "<tr><td>Net income</td><td>$456,789</td></tr>"
        "</table>"
    )
    chunkset = _chunkset_with_table(table_html)
    client = _fake_client_sequence(
        [_valid_narrative(), _valid_financials()]
    )
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-table-001",
        cache_root=tmp_path,
        anthropic_client=client,
        chunkset=chunkset,
    )
    assert out is not None
    assert out.financials is not None
    assert out.financials.revenue is not None
    assert out.financials.revenue.value_usd == 1234567.0
    assert out.financials.revenue.confidence == "high"
    assert out.financials.gross_profit is None
    assert out.financials.net_income is not None
    assert client.messages.create.call_count == 2  # type: ignore[attr-defined]


def test_render_table_html_strips_tags_and_bridges_cells() -> None:
    """The LLM-prompted quote 'Total revenue $1,234' cannot bridge
    `</td><td>` in raw HTML, but does in rendered text — so the worker
    MUST render before passing to the LLM."""
    import re

    from auto_research.extract.workers._common import _quote_to_flex_regex

    html = "<table><tr><td>Total revenue</td><td>$1,234</td></tr></table>"
    rendered = _render_table_html_to_text(html)
    assert "Total revenue" in rendered
    assert "$1,234" in rendered
    assert "<td>" not in rendered
    pattern = _quote_to_flex_regex("Total revenue $1,234")
    assert re.search(pattern, rendered) is not None


def test_merge_financials_takes_first_non_none_per_field() -> None:
    """Income statement provides revenue/net_income; balance sheet
    provides total_assets; cash flow provides cash_from_*. First
    non-None wins per field (document order ensures primary statements
    win over later notes-table sub-aggregations)."""

    def _line(value: float) -> FinancialLineItem:
        return FinancialLineItem(
            value_usd=value,
            citation=Citation(source_span=(0, 4), source_quote="hell"),
            confidence="high",
        )

    income = TenKFinancials.model_validate(
        {
            "revenue": _line(100.0).model_dump(),
            "gross_profit": None,
            "operating_income": None,
            "net_income": _line(20.0).model_dump(),
            "total_assets": None,
            "total_liabilities": None,
            "stockholders_equity": None,
            "cash_from_operations": None,
            "cash_from_investing": None,
            "cash_from_financing": None,
        }
    )
    balance = TenKFinancials.model_validate(
        {
            "revenue": None,
            "gross_profit": None,
            "operating_income": None,
            "net_income": None,
            "total_assets": _line(500.0).model_dump(),
            "total_liabilities": _line(300.0).model_dump(),
            "stockholders_equity": _line(200.0).model_dump(),
            "cash_from_operations": None,
            "cash_from_investing": None,
            "cash_from_financing": None,
        }
    )
    cashflow = TenKFinancials.model_validate(
        {
            "revenue": None,
            "gross_profit": None,
            "operating_income": None,
            "net_income": None,
            "total_assets": None,
            "total_liabilities": None,
            "stockholders_equity": None,
            "cash_from_operations": _line(50.0).model_dump(),
            "cash_from_investing": _line(-10.0).model_dump(),
            "cash_from_financing": _line(-5.0).model_dump(),
        }
    )
    merged = _merge_financials([income, balance, cashflow])
    assert merged.revenue is not None and merged.revenue.value_usd == 100.0
    assert merged.net_income is not None and merged.net_income.value_usd == 20.0
    assert merged.total_assets is not None and merged.total_assets.value_usd == 500.0
    assert merged.cash_from_operations is not None
    assert merged.cash_from_operations.value_usd == 50.0


def test_ten_k_item8_iterates_all_table_parents(tmp_path: Path) -> None:
    """Real 10-K Item 8 emits income statement, balance sheet, cash flow
    as separate <table> parents. Worker MUST run the financials prompt
    against each and merge — taking only [0] silently drops balance
    sheet + cash-flow line items."""
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-multi-table",
    )
    income_html = "<table><tr><td>Revenue</td><td>$100</td></tr></table>"
    balance_html = (
        "<table><tr><td>Total assets</td><td>$500</td></tr></table>"
    )
    parents = (
        ParentChunk(
            text=income_html,
            section_name="item_8",
            char_span=(0, len(income_html)),
            token_count=10,
            table_html=income_html,
            metadata=meta,
        ),
        ParentChunk(
            text=balance_html,
            section_name="item_8",
            char_span=(0, len(balance_html)),
            token_count=10,
            table_html=balance_html,
            metadata=meta,
        ),
    )
    chunkset = ChunkSet(parents=parents, children=())

    income_response = {
        "revenue": {
            "value_usd": 100.0,
            "citation": {"source_quote": "Revenue $100"},
            "confidence": "high",
        },
        "gross_profit": None,
        "operating_income": None,
        "net_income": None,
        "total_assets": None,
        "total_liabilities": None,
        "stockholders_equity": None,
        "cash_from_operations": None,
        "cash_from_investing": None,
        "cash_from_financing": None,
    }
    balance_response = {
        "revenue": None,
        "gross_profit": None,
        "operating_income": None,
        "net_income": None,
        "total_assets": {
            "value_usd": 500.0,
            "citation": {"source_quote": "Total assets $500"},
            "confidence": "high",
        },
        "total_liabilities": None,
        "stockholders_equity": None,
        "cash_from_operations": None,
        "cash_from_investing": None,
        "cash_from_financing": None,
    }
    client = _fake_client_sequence(
        [_valid_narrative(), income_response, balance_response]
    )
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-multi-table",
        cache_root=tmp_path,
        anthropic_client=client,
        chunkset=chunkset,
    )
    assert out is not None
    assert out.financials is not None
    assert out.financials.revenue is not None
    assert out.financials.revenue.value_usd == 100.0
    assert out.financials.total_assets is not None
    assert out.financials.total_assets.value_usd == 500.0
    # 1 narrative + 2 financials = 3 LLM calls.
    assert client.messages.create.call_count == 3  # type: ignore[attr-defined]


def test_ten_k_item8_financials_skipped_when_no_table_parents(
    tmp_path: Path,
) -> None:
    """Chunkset present but with no `table_html` parents → only the
    narrative call runs; financials stays None."""
    client = _fake_client_single(_valid_narrative())
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-no-table",
        cache_root=tmp_path,
        anthropic_client=client,
        chunkset=_chunkset_narrative_only(),
    )
    assert out is not None
    assert out.financials is None
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]


# --- Failure routing --------------------------------------------------------


def test_ten_k_hallucinated_quote_quarantines(tmp_path: Path) -> None:
    bad = _valid_narrative()
    bad["guidance_tone"]["citation"]["source_quote"] = "not in the filing at all"
    client = _fake_client_single(bad)
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-bad",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is None
    assert (tmp_path / "q" / "ten_k" / "10k-bad.json").exists()


def test_ten_k_item8_failure_does_not_drop_narrative(tmp_path: Path) -> None:
    """If the financials call quarantines, the worker returns the
    narrative output with `financials=None` — partial output is OK on
    the Item 8 path because the table is a side payload, not the
    primary signal."""
    table_html = "<table><tr><td>Revenue</td><td>$100</td></tr></table>"
    chunkset = _chunkset_with_table(table_html)
    bad_financials = _valid_financials()
    bad_financials["revenue"]["citation"]["source_quote"] = "not in the table"
    client = _fake_client_sequence(
        [_valid_narrative(), bad_financials]
    )
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-fin-bad",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
    )
    assert out is not None
    assert out.financials is None
    # Quarantine record exists for the financials call.
    assert (tmp_path / "q" / "ten_k" / "10k-fin-bad#item8.0.json").exists()


# --- Cache --------------------------------------------------------------


def test_ten_k_single_shot_cache_hit_skips_llm(tmp_path: Path) -> None:
    client = _fake_client_single(_valid_narrative())
    first = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-cache",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    second = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-cache",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert first == second
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]


# --- Citation-grounding on a real fixture (AC bullet 1) --------------------


def test_ten_k_real_fixture_passes_citation_grounding(tmp_path: Path) -> None:
    """End-to-end: realistic 10-K Item 7 excerpt + frozen LLM response →
    output passes citation grounding (no quarantine record written).
    Every Citation.source_quote indexes back to the raw text byte-exactly.
    AC bullet 1.
    """
    from auto_research.extract.guardrails import _walk_citations

    fixture_dir = Path(__file__).parent / "fixtures" / "ten_k"
    raw = (fixture_dir / "sample_item7.txt").read_text()
    frozen = json.loads((fixture_dir / "sample_item7_output.json").read_text())
    client = _fake_client_single(frozen)
    out = extract_ten_k(
        raw_doc=raw,
        doc_id="ten-k-fixture-001",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is not None
    # No quarantine record was written — the guardrail passed.
    quarantine_dir = tmp_path / "q"
    assert not quarantine_dir.exists() or not list(quarantine_dir.rglob("*.json"))
    # Every Citation.source_quote slices back to the raw text byte-exactly.
    for path, citation in _walk_citations(out):
        start, end = citation.source_span
        assert raw[start:end] == citation.source_quote, (
            f"mismatch at {path}: span=({start},{end}) "
            f"quote={citation.source_quote!r}"
        )
