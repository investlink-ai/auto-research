"""Unit tests for the S-1/S-3 extraction worker (Issue #11).

End-to-end of `extract_s_filing`: prompt → Anthropic → JSON parse →
SFilingOutput → citation grounding → cache. The Anthropic SDK is mocked
to keep the test hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from auto_research.extract.workers.s_filings import extract_s_filing

_SAMPLE_S3 = (
    "This shelf takedown of $200 million of common stock will be used for "
    "general corporate purposes and to fund the Phase II clinical trial."
)


def _make_response(body: dict[str, Any]) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(body), citations=None)],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=100,
            output_tokens=50,
            cache_creation=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


def _valid_output_for(text: str) -> dict[str, Any]:
    quote = "shelf takedown of $200 million of common stock"
    start = text.find(quote)
    end = start + len(quote)
    return {
        "cik": "0000000001",
        "accession_number": "0000000001-25-000001",
        "form_type": "S-3",
        "dilution_event": {
            "citation": {"source_span": [start, end], "source_quote": quote},
            "confidence": 0.9,
        },
        "capital_raise_language": [],
        "use_of_proceeds": [],
    }


def _fake_client(body: dict[str, Any]) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_response(body)
    return cast(anthropic.Anthropic, fake)


def test_extract_s_filing_returns_validated_output(tmp_path: Path) -> None:
    client = _fake_client(_valid_output_for(_SAMPLE_S3))
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="test-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.form_type == "S-3"
    assert out.dilution_event.confidence == pytest.approx(0.9)


def test_cache_hit_skips_llm_call(tmp_path: Path) -> None:
    """Second call with identical inputs must NOT touch the Anthropic SDK."""
    body = _valid_output_for(_SAMPLE_S3)
    client = _fake_client(body)
    first = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="test-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]
    second = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="test-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]
    assert first == second


def test_corrupted_citation_routes_to_quarantine(tmp_path: Path) -> None:
    """A hallucinated quote (not present verbatim in `raw_doc`) must
    route to quarantine. Spans are computed by the worker, so the way
    to simulate a hallucination is to corrupt the quote itself."""
    bad = _valid_output_for(_SAMPLE_S3)
    # Mutate the quote so it no longer appears in `_SAMPLE_S3`:
    bad["dilution_event"]["citation"]["source_quote"] = (
        "shelf takedown of $999 trillion of common stock"
    )
    client = _fake_client(bad)
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="bad-001",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quarantine",
        anthropic_client=client,
    )
    assert out is None
    qfile = tmp_path / "quarantine" / "s_filings" / "bad-001.json"
    assert qfile.exists()
