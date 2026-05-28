"""Unit tests for the 8-K extraction worker.

The shared scaffolding (`_common.py`) is exercised by its own test
module. The tests here are worker-specific: prompt + schema +
routing-table key wiring through `extract_eight_k`. Hermetic — the
Anthropic SDK is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto_research.extract.enums import EventClassification
from auto_research.extract.workers.eight_k import extract_eight_k
from tests.unit.conftest import make_fake_anthropic_client as _fake_client

_SAMPLE_8K = (
    "On January 15, 2026, the company entered into a Material Definitive "
    "Agreement with the Department of Defense for delivery of optical "
    "interconnect systems valued at $42 million."
)


def _valid_output() -> dict[str, Any]:
    return {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000007",
        "event_classification": "contract",
        "milestone_mentions": [],
        "dilution_language_flags": [
            {
                "citation": {"source_quote": "Material Definitive Agreement"},
                "confidence": "high",
            }
        ],
    }


def test_extract_eight_k_returns_validated_output(tmp_path: Path) -> None:
    client = _fake_client(_valid_output())
    out = extract_eight_k(
        raw_doc=_SAMPLE_8K,
        doc_id="8k-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.event_classification == EventClassification.CONTRACT
    assert len(out.dilution_language_flags) == 1


def test_extract_eight_k_resolves_span_into_raw_doc(tmp_path: Path) -> None:
    client = _fake_client(_valid_output())
    out = extract_eight_k(
        raw_doc=_SAMPLE_8K,
        doc_id="8k-002",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    citation = out.dilution_language_flags[0].citation
    start, end = citation.source_span
    assert _SAMPLE_8K[start:end] == citation.source_quote


def test_extract_eight_k_cache_hit_skips_llm(tmp_path: Path) -> None:
    client = _fake_client(_valid_output())
    first = extract_eight_k(
        raw_doc=_SAMPLE_8K,
        doc_id="8k-cache",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    second = extract_eight_k(
        raw_doc=_SAMPLE_8K,
        doc_id="8k-cache",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert first == second
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]


def test_extract_eight_k_quarantines_hallucinated_quote(tmp_path: Path) -> None:
    bad = _valid_output()
    bad["milestone_mentions"] = [
        {
            "citation": {"source_quote": "not in the filing at all"},
            "confidence": "high",
        }
    ]
    client = _fake_client(bad)
    out = extract_eight_k(
        raw_doc=_SAMPLE_8K,
        doc_id="8k-bad",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is None
    assert (tmp_path / "q" / "eight_k" / "8k-bad.json").exists()


def test_extract_eight_k_rejects_invalid_event_classification(tmp_path: Path) -> None:
    """Closed-set enum: an unknown classification must quarantine, not
    silently round-trip."""
    bad = _valid_output()
    bad["event_classification"] = "not-a-real-classification"
    client = _fake_client(bad)
    out = extract_eight_k(
        raw_doc=_SAMPLE_8K,
        doc_id="8k-enum",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is None
    record = json.loads((tmp_path / "q" / "eight_k" / "8k-enum.json").read_text())
    assert "schema validation failed" in record["error"]


# --- Citation-grounding on a real fixture (AC bullet 1) --------------------


def test_eight_k_real_fixture_passes_citation_grounding(tmp_path: Path) -> None:
    """End-to-end: realistic 8-K excerpt + frozen LLM response → output
    passes citation grounding. Every Citation.source_quote indexes back
    to the raw text byte-exactly. AC bullet 1.
    """
    from auto_research.extract.guardrails import _walk_citations

    fixture_dir = Path(__file__).parent / "fixtures" / "eight_k"
    raw = (fixture_dir / "sample_8k.txt").read_text()
    frozen = json.loads((fixture_dir / "sample_8k_output.json").read_text())
    client = _fake_client(frozen)
    out = extract_eight_k(
        raw_doc=raw,
        doc_id="8k-fixture-001",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is not None
    quarantine_dir = tmp_path / "q"
    assert not quarantine_dir.exists() or not list(quarantine_dir.rglob("*.json"))
    for path, citation in _walk_citations(out):
        start, end = citation.source_span
        assert raw[start:end] == citation.source_quote, (
            f"mismatch at {path}: span=({start},{end}) "
            f"quote={citation.source_quote!r}"
        )
