"""Unit tests for the S-1/S-3 extraction worker.

End-to-end of `extract_s_filing`: prompt -> Anthropic tool_use ->
SFilingOutput -> citation grounding -> cache. The Anthropic SDK is mocked
to keep the test hermetic.

Coverage focus per the PR's code-review pass:
- Cache hit skips the SDK call entirely.
- Hallucinated quote (not findable in raw) -> quarantine.
- Ambiguous quote (multiple matches) -> quarantine.
- Empty quote -> quarantine (not uncaught ValidationError).
- Schema violation (extra field, wrong shape) -> quarantine.
- QuarantineRecord captures the original parsed dict, not the worker's
  mutated copy.
- Resolved source_span aligns with raw_doc, not a normalized form.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, ToolUseBlock, Usage

from auto_research.extract.workers.s_filings import extract_s_filing
from tests._otel_helpers import SpanRecorder

# Two-line raw doc so we exercise whitespace-flexible matching across a
# newline that the LLM would naturally collapse when quoting.
_SAMPLE_S3 = (
    "This shelf takedown of $200 million of common stock\n"
    "will be used for general corporate purposes and to fund\n"
    "the Phase II clinical trial."
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
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="tool_use",
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


def _valid_output() -> dict[str, Any]:
    return {
        "cik": "0000000001",
        "accession_number": "0000000001-25-000001",
        "form_type": "S-3",
        "dilution_event": {
            # Whitespace-collapsed quote — exercises the flexible-regex
            # path. The substring is unique in the raw doc.
            "citation": {
                "source_quote": "shelf takedown of $200 million of common stock"
            },
            "confidence": "high",
        },
        "capital_raise_language": [],
        "use_of_proceeds": [],
    }


def _fake_client(tool_input: Any) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_tool_response(tool_input)
    return cast(anthropic.Anthropic, fake)


def test_extract_s_filing_returns_validated_output(tmp_path: Path) -> None:
    client = _fake_client(_valid_output())
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="test-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.form_type == "S-3"
    assert out.dilution_event.confidence == "high"


def test_resolved_span_indexes_into_raw_doc(tmp_path: Path) -> None:
    """Citation.source_span must index into `raw_doc` (not a normalized
    form); slicing raw with the span must equal source_quote."""
    client = _fake_client(_valid_output())
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="test-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    cite = out.dilution_event.citation
    start, end = cite.source_span
    assert _SAMPLE_S3[start:end] == cite.source_quote


def test_cache_hit_skips_llm_call(tmp_path: Path) -> None:
    client = _fake_client(_valid_output())
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


def test_hallucinated_quote_routes_to_quarantine(tmp_path: Path) -> None:
    bad = _valid_output()
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


def test_ambiguous_quote_routes_to_quarantine(tmp_path: Path) -> None:
    """A quote that appears multiple times in raw must quarantine — the
    worker cannot honestly pick one location over another."""
    raw = "general corporate purposes. ... general corporate purposes."
    bad = _valid_output()
    bad["dilution_event"]["citation"]["source_quote"] = "general corporate purposes"
    client = _fake_client(bad)
    out = extract_s_filing(
        raw_doc=raw,
        doc_id="amb-001",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quarantine",
        anthropic_client=client,
    )
    assert out is None
    qfile = tmp_path / "quarantine" / "s_filings" / "amb-001.json"
    assert qfile.exists()
    record = json.loads(qfile.read_text())
    assert "AMBIGUOUS" in record["error"]


def test_empty_quote_routes_to_quarantine(tmp_path: Path) -> None:
    """Empty source_quote must quarantine — used to crash with
    `start < end` ValidationError because raw.find('') returned 0."""
    bad = _valid_output()
    bad["dilution_event"]["citation"]["source_quote"] = ""
    client = _fake_client(bad)
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="empty-001",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quarantine",
        anthropic_client=client,
    )
    assert out is None
    assert (tmp_path / "quarantine" / "s_filings" / "empty-001.json").exists()


def test_schema_violation_routes_to_quarantine(tmp_path: Path) -> None:
    """An extra/invalid field must quarantine, not crash with ValidationError."""
    bad = _valid_output()
    bad["unexpected_field"] = "boom"
    client = _fake_client(bad)
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="schema-001",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quarantine",
        anthropic_client=client,
    )
    assert out is None
    qfile = tmp_path / "quarantine" / "s_filings" / "schema-001.json"
    assert qfile.exists()
    record = json.loads(qfile.read_text())
    assert "schema validation failed" in record["error"]


def test_production_client_is_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production extractions (no injected client) reuse one
    `make_extraction_client` instance so the per-worker @cost_cap and
    @circuit_breaker state accumulates across calls — see
    `src/auto_research/extract/client.py` lines 39, 96-97 ('Production
    code instantiates one client per worker module ... at module top
    level so the per-worker budgets are independent')."""
    from auto_research.extract.client import make_extraction_client as real_factory
    from auto_research.extract.workers import s_filings as worker_mod

    monkeypatch.setattr(worker_mod, "_CLIENT", None)
    factory_calls = 0

    def counting_factory(**kwargs: Any) -> Any:
        nonlocal factory_calls
        factory_calls += 1
        # Inject a never-firing anthropic client so the real factory does
        # not try to read ANTHROPIC_API_KEY from env.
        return real_factory(anthropic_client=cast(anthropic.Anthropic, MagicMock()), **kwargs)

    monkeypatch.setattr(worker_mod, "make_extraction_client", counting_factory)

    a = worker_mod._get_client(None)
    b = worker_mod._get_client(None)
    assert a is b, "production path must return the same client instance"
    assert factory_calls == 1, "factory must build exactly one production client"


def test_injected_client_bypasses_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test-injection path gets a fresh client per call — required for
    hermetic per-test state, and explicitly NOT what production wants."""
    from auto_research.extract.workers import s_filings as worker_mod

    monkeypatch.setattr(worker_mod, "_CLIENT", None)
    fake1 = cast(anthropic.Anthropic, MagicMock())
    fake2 = cast(anthropic.Anthropic, MagicMock())
    a = worker_mod._get_client(fake1)
    b = worker_mod._get_client(fake2)
    assert a is not b


def test_quarantine_captures_original_parsed_not_mutated(tmp_path: Path) -> None:
    """QuarantineRecord must show what the LLM returned, not the worker's
    mutated copy with sentinel spans."""
    bad = _valid_output()
    bad["dilution_event"]["citation"]["source_quote"] = "not-in-doc"
    # Intentionally include source_span as the model might (forbidden but
    # tolerated for audit purposes); the snapshot should preserve it.
    bad["dilution_event"]["citation"]["source_span"] = [99, 100]
    client = _fake_client(bad)
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="audit-001",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quarantine",
        anthropic_client=client,
    )
    assert out is None
    record = json.loads(
        (tmp_path / "quarantine" / "s_filings" / "audit-001.json").read_text()
    )
    captured_citation = record["output"]["dilution_event"]["citation"]
    assert captured_citation["source_quote"] == "not-in-doc"
    # The model's original span survives — not a sentinel injected by the worker
    assert captured_citation["source_span"] == [99, 100]


# ---------- OTel instrumentation (refs #52) ----------


def test_extract_s_filing_emits_span_persisted(
    span_recorder: SpanRecorder, tmp_path: Path
) -> None:
    """Successful extraction → outcome=persisted (parents the existing
    llm.cost.est_usd attribute set by extract/client.py:151)."""
    client = _fake_client(_valid_output())
    extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="doc-persist",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    attrs = span_recorder.attrs("extract.s_filings")
    assert attrs["extract.worker"] == "s_filings"
    assert attrs["extract.doc_id"] == "doc-persist"
    assert attrs["extract.outcome"] == "persisted"


def test_extract_s_filing_emits_span_cache_hit(
    span_recorder: SpanRecorder, tmp_path: Path
) -> None:
    """A second invocation should record outcome=cache_hit."""
    client = _fake_client(_valid_output())
    # Seed the cache via a first call.
    extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="doc-cache",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    # Second call should hit cache without invoking the LLM.
    extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="doc-cache",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    spans = span_recorder.by_name("extract.s_filings")
    assert len(spans) == 2
    assert spans[0].attributes is not None
    assert spans[1].attributes is not None
    assert spans[0].attributes["extract.outcome"] == "persisted"
    assert spans[1].attributes["extract.outcome"] == "cache_hit"


def test_extract_s_filing_emits_span_quarantined_on_schema_violation(
    span_recorder: SpanRecorder, tmp_path: Path
) -> None:
    """A tool_use.input that fails pydantic validation → outcome=quarantined."""
    bad = _valid_output()
    bad["unexpected_field"] = "boom"  # extra='forbid' rejects this
    client = _fake_client(bad)
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="doc-bad-schema",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quar",
        anthropic_client=client,
    )
    assert out is None
    attrs = span_recorder.attrs("extract.s_filings")
    assert attrs["extract.outcome"] == "quarantined"
    # Every quarantine branch must also set span.status=ERROR so
    # alerting wired against OTel status surfaces guardrail failures
    # (INV-2).
    from opentelemetry.trace import StatusCode

    span = span_recorder.one("extract.s_filings")
    assert span.status.status_code == StatusCode.ERROR


def test_extract_s_filing_emits_span_error_on_client_raise(
    span_recorder: SpanRecorder, tmp_path: Path
) -> None:
    """When the Anthropic client raises (network / 429 / cost cap),
    the span must record outcome='error' AND status=ERROR so the
    documented enum stays complete on infra-failure paths."""
    from opentelemetry.trace import StatusCode

    fake = MagicMock()
    fake.messages.create.side_effect = RuntimeError("simulated 503")
    client = cast(anthropic.Anthropic, fake)
    with pytest.raises(RuntimeError):
        extract_s_filing(
            raw_doc=_SAMPLE_S3,
            doc_id="doc-client-raise",
            cache_root=tmp_path,
            quarantine_root=tmp_path / "quar",
            anthropic_client=client,
        )
    attrs = span_recorder.attrs("extract.s_filings")
    assert attrs["extract.outcome"] == "error"
    span = span_recorder.one("extract.s_filings")
    assert span.status.status_code == StatusCode.ERROR
