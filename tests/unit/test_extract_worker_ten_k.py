"""Unit tests for the 10-K extraction worker.

Hermetic — the Anthropic SDK is mocked. Branch-coverage tests satisfy
AC bullet 3 ("Hybrid extraction policy: single-shot for
< SINGLE_SHOT_TOKEN_CUTOFF tokens, RAG path for ≥").

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
    ChunkMetadata,
    ChunkSet,
    ParentChunk,
)
from auto_research.extract.workers.ten_k import extract_ten_k
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
        "going_concern": None,
        "icfr_material_weaknesses": [],
        "critical_accounting_estimate_changes": [],
    }


def _chunkset_narrative_only() -> ChunkSet:
    """A minimal chunkset — exercises the RAG narrative branch code path."""
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
    """Short raw doc, no chunkset → single-shot. Exactly one LLM call."""
    client = _fake_client_single(_valid_narrative())
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.fiscal_period_end == date(2025, 12, 31)
    assert out.supplier_mentions[0].mention_text == "TSMC"
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]


def test_ten_k_single_shot_branch_short_doc_with_narrative_chunkset(
    tmp_path: Path,
) -> None:
    """Short raw doc + chunkset (no table parents) → still single-shot
    narrative (chunkset alone does NOT trigger the RAG path; the size
    threshold does)."""
    client = _fake_client_single(_valid_narrative())
    out = extract_ten_k(
        raw_doc=_SAMPLE_10K,
        doc_id="10k-002",
        cache_root=tmp_path,
        anthropic_client=client,
        chunkset=_chunkset_narrative_only(),
    )
    assert out is not None
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
        "going_concern": "going concern",
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
        if field == "going_concern":
            return {
                **base,
                "going_concern": {
                    "citation": {"source_quote": sentinel},
                    "confidence": "high",
                },
            }
        raise ValueError(f"unknown field {field!r}")

    # The worker iterates `TEN_K_NARRATIVE_FIELD_CONFIGS` in order:
    # guidance_tone, accrual_flags, supplier_mentions,
    # customer_mentions, risk_factor_deltas, going_concern.
    responses_in_order = [
        _response_for(f)
        for f in (
            "guidance_tone",
            "accrual_flags",
            "supplier_mentions",
            "customer_mentions",
            "risk_factor_deltas",
            "going_concern",
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
    assert len(queries_seen) == 6  # one query per narrative field
    assert client.messages.create.call_count == 6  # type: ignore[attr-defined]


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
    """Six partial-schema dicts (one per narrative field) whose
    citation quote ('cautious growth in fiscal 2026') is present in
    `_chunkset_narrative_only`'s parent text — so each per-field
    response validates cleanly against its respective Pydantic partial.

    Order matches `TEN_K_NARRATIVE_FIELD_CONFIGS`: guidance_tone,
    accrual_flags, supplier_mentions, customer_mentions,
    risk_factor_deltas, going_concern.
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
        {
            **base,
            "going_concern": {
                "citation": {"source_quote": quote},
                "confidence": "high",
            },
        },
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
    # Exactly 3 SDK calls were attempted (2 successes + the failing
    # supplier_mentions call). Without this assertion, a regression
    # that short-circuited earlier would still see `out is None` +
    # empty cache and pass.
    assert client.messages.create.call_count == 3  # type: ignore[attr-defined]
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
    # diverged cik. The remaining 5 keep the agreed cik.
    client = _fake_client_sequence(
        [base[0], diverged[1], base[2], base[3], base[4], base[5]]
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
    # All 6 per-field calls were attempted before the identity check
    # fired. Without this, a regression that short-circuited on the
    # first non-matching cik would still see `out is None` + empty
    # cache and pass.
    assert client.messages.create.call_count == 6  # type: ignore[attr-defined]
    # Identity check fires BEFORE commit, so staged writes are dropped
    # and the cache stays empty even though all 6 calls validated.
    assert list((tmp_path / "cache").rglob("*.json")) == []


def test_ten_k_rag_populates_going_concern_when_planted(
    tmp_path: Path,
) -> None:
    """When the retrieved Item 8 audit passage contains a substantial-
    doubt sentence, the going_concern field on the merged TenKOutput
    is a Claim quoting that sentence."""
    long_raw = "word " * 200_000
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-going-001",
    )
    going_concern_text = (
        "the conditions raise substantial doubt about the Company's "
        "ability to continue as a going concern."
    )
    parent_text = f"Item 8 audit report. {going_concern_text}"
    parent = ParentChunk(
        text=parent_text,
        section_name="item_8",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    base = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    responses = [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": "audit report"},
                "confidence": "medium",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
        {
            **base,
            "going_concern": {
                "citation": {"source_quote": going_concern_text},
                "confidence": "high",
            },
        },
    ]
    client = _fake_client_sequence(responses)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-going-001",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: [parent],
    )
    assert out is not None
    assert out.going_concern is not None
    assert "substantial doubt" in out.going_concern.citation.source_quote
    assert out.going_concern.confidence == "high"
    assert client.messages.create.call_count == 6  # type: ignore[attr-defined]


def test_ten_k_rag_going_concern_absent_returns_none(
    tmp_path: Path,
) -> None:
    """Unqualified audit opinion → partial returns going_concern=None
    → merged TenKOutput carries None."""
    long_raw = "word " * 200_000
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-going-002",
    )
    parent_text = (
        "Item 8 audit report. In our opinion, the financial statements "
        "present fairly, in all material respects, the financial "
        "position of the Company."
    )
    parent = ParentChunk(
        text=parent_text,
        section_name="item_8",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    base = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    responses = [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": "Company"},
                "confidence": "low",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
        {**base, "going_concern": None},
    ]
    client = _fake_client_sequence(responses)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-going-002",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: [parent],
    )
    assert out is not None
    assert out.going_concern is None


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
