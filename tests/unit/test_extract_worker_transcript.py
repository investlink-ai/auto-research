"""Unit tests for the transcript extraction worker.

Hermetic — the Anthropic SDK is mocked. The shared scaffolding
(`_common.py`) is exercised by its own test module; the tests here
verify the binary-split prompt + partial schemas + routing-table key
wire up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto_research.extract.workers.transcript import extract_transcript
from tests.unit.conftest import (
    make_fake_anthropic_client_sequence as _fake_client_sequence,
)

_SAMPLE_TRANSCRIPT = (
    "Operator: Welcome to ACME's fiscal 2026 first-quarter earnings call.\n"
    "CFO: We had a strong quarter with revenue up 30% year over year.\n"
    "Q&A Analyst: What about FY26 gross margins?\n"
    "CFO: we can't comment on that beyond what's in our guidance.\n"
)


def _valid_prepared() -> dict[str, Any]:
    return {
        "ticker": "ACME",
        "event_datetime": "2026-01-30T17:00:00-05:00",
        "prepared_remarks_tone": {
            "citation": {
                "source_quote": "We had a strong quarter with revenue up 30%"
            },
            "confidence": "high",
        },
    }


def _valid_qa() -> dict[str, Any]:
    return {
        "ticker": "ACME",
        "event_datetime": "2026-01-30T17:00:00-05:00",
        "q_and_a_evasiveness": {
            "citation": {
                "source_quote": "we can't comment on that beyond what's in our guidance"
            },
            "confidence": "medium",
        },
        "forward_statements": [],
    }


def test_extract_transcript_returns_validated_output(tmp_path: Path) -> None:
    client = _fake_client_sequence([_valid_prepared(), _valid_qa()])
    out = extract_transcript(
        raw_doc=_SAMPLE_TRANSCRIPT,
        doc_id="trn-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.ticker == "ACME"
    assert out.event_datetime is not None
    assert out.event_datetime.year == 2026
    assert out.prepared_remarks_tone.confidence == "high"
    assert out.q_and_a_evasiveness.confidence == "medium"
    assert client.messages.create.call_count == 2  # type: ignore[attr-defined]


def test_extract_transcript_resolves_spans_into_raw_doc(tmp_path: Path) -> None:
    client = _fake_client_sequence([_valid_prepared(), _valid_qa()])
    out = extract_transcript(
        raw_doc=_SAMPLE_TRANSCRIPT,
        doc_id="trn-002",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    for citation in (
        out.prepared_remarks_tone.citation,
        out.q_and_a_evasiveness.citation,
    ):
        start, end = citation.source_span
        assert _SAMPLE_TRANSCRIPT[start:end] == citation.source_quote


def test_extract_transcript_cache_hit_skips_llm(tmp_path: Path) -> None:
    """After the first invocation commits both partials' cache entries,
    a second invocation must hit the cache for both calls and skip the
    LLM entirely. Total SDK calls across two invocations is 2 (both
    from the first run)."""
    client = _fake_client_sequence([_valid_prepared(), _valid_qa()])
    extract_transcript(
        raw_doc=_SAMPLE_TRANSCRIPT,
        doc_id="trn-cache",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    extract_transcript(
        raw_doc=_SAMPLE_TRANSCRIPT,
        doc_id="trn-cache",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert client.messages.create.call_count == 2  # type: ignore[attr-defined]


def test_extract_transcript_quarantines_hallucinated_quote(tmp_path: Path) -> None:
    """A hallucinated source_quote on the QA call quarantines the whole
    transcript — the prepared-remarks call's cache write is dropped."""
    bad_qa = _valid_qa()
    bad_qa["forward_statements"] = [
        {
            "statement_text": "FY27 revenue will double",
            "citation": {"source_quote": "FY27 revenue will double"},
            "mentioned_entities": [],
            "horizon": "long-term",
        }
    ]
    client = _fake_client_sequence([_valid_prepared(), bad_qa])
    out = extract_transcript(
        raw_doc=_SAMPLE_TRANSCRIPT,
        doc_id="trn-bad",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is None
    assert (tmp_path / "q" / "transcript" / "trn-bad#qa.json").exists()


def test_extract_transcript_quarantines_on_identity_disagreement(
    tmp_path: Path,
) -> None:
    """If the prepared-remarks and Q&A calls return different `ticker`,
    the worker MUST quarantine rather than silently picking one."""
    diverged = _valid_qa()
    diverged["ticker"] = "ZZZZ"  # disagrees with prepared's "ACME"
    client = _fake_client_sequence([_valid_prepared(), diverged])
    out = extract_transcript(
        raw_doc=_SAMPLE_TRANSCRIPT,
        doc_id="trn-identity",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is None
    qrec = tmp_path / "q" / "transcript" / "trn-identity#identity-disagreement.json"
    assert qrec.exists()
    record = json.loads(qrec.read_text())
    assert "ticker" in record["error"]
    # Identity check fires BEFORE commit, so staged writes are dropped
    # and the cache stays empty.
    assert list((tmp_path / "cache").rglob("*.json")) == []


# --- Citation-grounding on a real fixture (AC bullet 1) --------------------


def test_transcript_real_fixture_passes_citation_grounding(tmp_path: Path) -> None:
    """End-to-end: realistic earnings-call transcript + frozen LLM
    responses → output passes citation grounding. Every
    Citation.source_quote indexes back to the raw text byte-exactly.
    AC bullet 1.

    The frozen fixture is the full TranscriptOutput shape; this test
    splits it into the two partial shapes the binary-split worker now
    expects.
    """
    from auto_research.extract.guardrails import _walk_citations

    fixture_dir = Path(__file__).parent / "fixtures" / "transcript"
    raw = (fixture_dir / "sample_transcript.txt").read_text()
    frozen = json.loads((fixture_dir / "sample_transcript_output.json").read_text())
    prepared_partial = {
        "ticker": frozen["ticker"],
        "event_datetime": frozen["event_datetime"],
        "prepared_remarks_tone": frozen["prepared_remarks_tone"],
    }
    qa_partial = {
        "ticker": frozen["ticker"],
        "event_datetime": frozen["event_datetime"],
        "q_and_a_evasiveness": frozen["q_and_a_evasiveness"],
        "forward_statements": frozen["forward_statements"],
    }
    client = _fake_client_sequence([prepared_partial, qa_partial])
    out = extract_transcript(
        raw_doc=raw,
        doc_id="trn-fixture-001",
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
