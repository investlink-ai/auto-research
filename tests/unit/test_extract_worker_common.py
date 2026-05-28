"""Unit tests for the shared extraction-worker scaffolding."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from auto_research.extract.enums import EventClassification
from auto_research.extract.schemas import EightKOutput
from auto_research.extract.workers._common import (
    _quote_to_flex_regex,
    _resolve_spans,
    _strip_fence,
    _write_quarantine,
    run_single_shot_extraction,
)


def test_strip_fence_removes_json_fence_with_newlines() -> None:
    assert _strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fence_removes_json_fence_no_newlines() -> None:
    assert _strip_fence('```{"a": 1}```') == '{"a": 1}'


def test_strip_fence_passthrough_when_no_fence() -> None:
    body = '{"a": 1}'
    assert _strip_fence(body) == body


def test_quote_to_flex_regex_collapses_whitespace() -> None:
    pattern = _quote_to_flex_regex("foo  bar")
    assert re.search(pattern, "foo\nbar") is not None


def test_quote_to_flex_regex_empty_quote_never_matches() -> None:
    pattern = _quote_to_flex_regex("")
    assert re.search(pattern, "any text") is None


def test_resolve_spans_finds_unique_quote() -> None:
    parsed = {
        "claim": {
            "citation": {"source_quote": "hello\nworld"},
            "confidence": 0.5,
        }
    }
    raw = "hello world is here"
    resolved, problems = _resolve_spans(parsed, raw)
    assert problems == []
    citation = resolved["claim"]["citation"]
    start, end = citation["source_span"]
    assert raw[start:end] == "hello world"
    assert citation["source_quote"] == "hello world"


def test_resolve_spans_flags_not_found() -> None:
    parsed: dict[str, Any] = {"citation": {"source_quote": "missing-quote"}}
    raw = "different text entirely"
    _, problems = _resolve_spans(parsed, raw)
    assert problems == ["missing-quote"]


def test_resolve_spans_flags_ambiguous_with_count() -> None:
    parsed: dict[str, Any] = {"citation": {"source_quote": "same"}}
    raw = "same same same"
    _, problems = _resolve_spans(parsed, raw)
    assert len(problems) == 1
    assert "AMBIGUOUS" in problems[0]
    assert "3 matches" in problems[0]


def test_resolve_spans_assigns_multiple_matches_in_document_order() -> None:
    """A real 10-K names TSMC across multiple sections and the model emits
    one SupplierMention per textual occurrence. When N citations share
    the same source_quote text and raw has exactly N occurrences, pair
    them in document order rather than rejecting all as AMBIGUOUS."""
    parsed: dict[str, Any] = {
        "supplier_mentions": [
            {"source_quote": "TSMC"},
            {"source_quote": "TSMC"},
            {"source_quote": "TSMC"},
        ]
    }
    raw = "Risk: TSMC supply. MD&A: TSMC pricing. Properties: TSMC fab 21."
    resolved, problems = _resolve_spans(parsed, raw)
    assert problems == []
    sm = resolved["supplier_mentions"]
    starts = [entry["source_span"][0] for entry in sm]
    assert starts == sorted(starts)
    for entry in sm:
        start, end = entry["source_span"]
        assert raw[start:end] == "TSMC"


def test_resolve_spans_flags_insufficient_matches_when_citations_exceed_occurrences() -> None:
    """N citations sharing a quote but raw has fewer occurrences →
    model fabricated extras; quarantine."""
    parsed: dict[str, Any] = {
        "supplier_mentions": [
            {"source_quote": "TSMC"},
            {"source_quote": "TSMC"},
        ]
    }
    raw = "Only one TSMC mention here."
    _, problems = _resolve_spans(parsed, raw)
    assert len(problems) == 1
    assert "INSUFFICIENT" in problems[0]


def test_resolve_spans_does_not_collide_distinct_quotes() -> None:
    """Two distinct quote strings do not get merged into one quote_to_nodes
    bucket even when one is a substring of the other — the dict key is
    the full quote string, so 'TSMC' and 'TSMC supply' are tracked
    independently."""
    parsed: dict[str, Any] = {
        "a": {"source_quote": "TSMC"},
        "b": {"source_quote": "TSMC supply"},
    }
    raw = "TSMC and TSMC supply mentioned here."
    resolved, problems = _resolve_spans(parsed, raw)
    # "TSMC" alone has 2 occurrences (positions 0 and 9), one citation
    # → AMBIGUOUS for that bucket; "TSMC supply" has 1 occurrence, one
    # citation → resolves cleanly.
    assert any("AMBIGUOUS" in p for p in problems)
    assert resolved["b"]["source_span"][0] == raw.index("TSMC supply")


def test_resolve_spans_does_not_mutate_input() -> None:
    parsed: dict[str, Any] = {"citation": {"source_quote": "hello"}}
    raw = "hello"
    snapshot = copy.deepcopy(parsed)
    _resolve_spans(parsed, raw)
    assert parsed == snapshot


def test_write_quarantine_writes_record(tmp_path: Path) -> None:
    _write_quarantine(
        quarantine_root=tmp_path / "q",
        worker="test_worker",
        prompt_version="v1",
        doc_id="doc-1",
        parsed={"raw": "thing"},
        error="bad json",
    )
    target = tmp_path / "q" / "test_worker" / "doc-1.json"
    assert target.exists()
    record = json.loads(target.read_text())
    assert record["worker"] == "test_worker"
    assert record["doc_id"] == "doc-1"
    assert record["error"] == "bad json"
    assert record["output"] == {"raw": "thing"}


def _make_response(text: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=text, citations=None)],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
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


def _fake_client(text: str) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_response(text)
    return cast(anthropic.Anthropic, fake)


def test_run_single_shot_extraction_happy_path(tmp_path: Path) -> None:
    raw = "Material agreement signed with the Department of Defense."
    payload = {
        "cik": "0000000001",
        "accession_number": "0000000001-25-000001",
        "event_classification": "contract",
        "milestone_mentions": [],
        "dilution_language_flags": [],
    }
    client = _fake_client(json.dumps(payload))
    out = run_single_shot_extraction(
        raw_doc=raw,
        doc_id="doc-1",
        worker="eight_k",
        task="event_classification",
        prompt="prompt-placeholder",
        prompt_version="v1",
        output_model=EightKOutput,
        max_tokens=512,
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    assert out is not None
    assert out.event_classification == EventClassification.CONTRACT


def test_run_single_shot_extraction_quarantines_bad_json(tmp_path: Path) -> None:
    client = _fake_client("not json")
    out = run_single_shot_extraction(
        raw_doc="x",
        doc_id="doc-bad",
        worker="eight_k",
        task="event_classification",
        prompt="prompt-placeholder",
        prompt_version="v1",
        output_model=EightKOutput,
        max_tokens=64,
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    assert out is None
    assert (tmp_path / "quar" / "eight_k" / "doc-bad.json").exists()


def test_run_single_shot_extraction_cache_hit_skips_llm(tmp_path: Path) -> None:
    raw = "Material agreement signed with the Department of Defense."
    payload = {
        "cik": "0000000001",
        "accession_number": "0000000001-25-000001",
        "event_classification": "contract",
        "milestone_mentions": [],
        "dilution_language_flags": [],
    }
    client = _fake_client(json.dumps(payload))
    first = run_single_shot_extraction(
        raw_doc=raw,
        doc_id="doc-cache",
        worker="eight_k",
        task="event_classification",
        prompt="prompt-placeholder",
        prompt_version="v1",
        output_model=EightKOutput,
        max_tokens=512,
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    second = run_single_shot_extraction(
        raw_doc=raw,
        doc_id="doc-cache",
        worker="eight_k",
        task="event_classification",
        prompt="prompt-placeholder",
        prompt_version="v1",
        output_model=EightKOutput,
        max_tokens=512,
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    assert first == second
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]


def test_run_single_shot_extraction_propagates_client_exception(tmp_path: Path) -> None:
    fake = MagicMock()
    fake.messages.create.side_effect = RuntimeError("simulated 503")
    client = cast(anthropic.Anthropic, fake)
    with pytest.raises(RuntimeError):
        run_single_shot_extraction(
            raw_doc="x",
            doc_id="doc-raise",
            worker="eight_k",
            task="event_classification",
            prompt="prompt-placeholder",
            prompt_version="v1",
            output_model=EightKOutput,
            max_tokens=64,
            cache_root=tmp_path / "cache",
            quarantine_root=tmp_path / "quar",
            anthropic_client=client,
        )
