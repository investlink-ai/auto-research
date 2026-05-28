"""Unit tests for the transcript extraction worker.

Hermetic — the Anthropic SDK is mocked. The shared scaffolding
(`_common.py`) is exercised by its own test module; the tests here
verify the transcript prompt + schema + routing-table key wire up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import anthropic
from anthropic.types import Message, ToolUseBlock, Usage

from auto_research.extract.workers.transcript import extract_transcript

_SAMPLE_TRANSCRIPT = (
    "Operator: Welcome to ACME's fiscal 2026 first-quarter earnings call.\n"
    "CFO: We had a strong quarter with revenue up 30% year over year.\n"
    "Q&A Analyst: What about FY26 gross margins?\n"
    "CFO: we can't comment on that beyond what's in our guidance.\n"
)


def _make_tool_response(tool_input: Any) -> Message:
    return Message(
        id="msg_test",
        content=[
            ToolUseBlock(
                id="toolu_test",
                input=tool_input,
                name="record_extraction",
                type="tool_use",
            )
        ],
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=10,
            output_tokens=10,
            cache_creation=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


def _fake_client(tool_input: Any) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_tool_response(tool_input)
    return cast(anthropic.Anthropic, fake)


def _valid_output() -> dict[str, Any]:
    return {
        "ticker": "ACME",
        "event_datetime": "2026-01-30T17:00:00-05:00",
        "prepared_remarks_tone": {
            "citation": {
                "source_quote": "We had a strong quarter with revenue up 30%"
            },
            "confidence": "high",
        },
        "q_and_a_evasiveness": {
            "citation": {
                "source_quote": "we can't comment on that beyond what's in our guidance"
            },
            "confidence": "medium",
        },
        "forward_statements": [],
    }


def test_extract_transcript_returns_validated_output(tmp_path: Path) -> None:
    client = _fake_client(_valid_output())
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


def test_extract_transcript_resolves_spans_into_raw_doc(tmp_path: Path) -> None:
    client = _fake_client(_valid_output())
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
    client = _fake_client(_valid_output())
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
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]


def test_extract_transcript_quarantines_hallucinated_quote(tmp_path: Path) -> None:
    bad = _valid_output()
    bad["forward_statements"] = [
        {
            "statement_text": "FY27 revenue will double",
            "citation": {"source_quote": "FY27 revenue will double"},
            "mentioned_entities": [],
            "horizon": "long-term",
        }
    ]
    client = _fake_client(bad)
    out = extract_transcript(
        raw_doc=_SAMPLE_TRANSCRIPT,
        doc_id="trn-bad",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
    )
    assert out is None
    assert (tmp_path / "q" / "transcript" / "trn-bad.json").exists()


# --- Citation-grounding on a real fixture (AC bullet 1) --------------------


def test_transcript_real_fixture_passes_citation_grounding(tmp_path: Path) -> None:
    """End-to-end: realistic earnings-call transcript + frozen LLM
    response → output passes citation grounding. Every
    Citation.source_quote indexes back to the raw text byte-exactly.
    AC bullet 1.
    """
    from auto_research.extract.guardrails import _walk_citations

    fixture_dir = Path(__file__).parent / "fixtures" / "transcript"
    raw = (fixture_dir / "sample_transcript.txt").read_text()
    frozen = json.loads((fixture_dir / "sample_transcript_output.json").read_text())
    client = _fake_client(frozen)
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
