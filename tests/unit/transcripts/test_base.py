"""Unit tests for `auto_research.ingest.transcripts._base`."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from auto_research.ingest.transcripts._base import (
    AudioSource,
    Transcript,
    TranscriptConfigError,
)


def _make_transcript(**overrides: object) -> Transcript:
    defaults: dict[str, object] = {
        "ticker": "NVDA",
        "year": 2024,
        "quarter": 2,
        "event_datetime": datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
        "prepared_remarks": "Thanks. Q2 revenue was $30B.",
        "q_and_a": "Operator: Our first question…",
    }
    defaults.update(overrides)
    return Transcript(**defaults)  # type: ignore[arg-type]


def test_transcript_constructs_with_all_fields() -> None:
    t = _make_transcript()
    assert t.ticker == "NVDA"
    assert t.year == 2024
    assert t.quarter == 2
    assert t.prepared_remarks.startswith("Thanks.")
    assert t.q_and_a.startswith("Operator:")


def test_transcript_is_frozen() -> None:
    t = _make_transcript()
    with pytest.raises(ValidationError):
        t.ticker = "AAPL"


def test_transcript_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Transcript(  # type: ignore[call-arg]
            ticker="NVDA",
            year=2024,
            quarter=2,
            event_datetime=datetime(2024, 5, 22, tzinfo=UTC),
            prepared_remarks="",
            q_and_a="",
            speaker_count=4,  # not in schema
        )


def test_transcript_validates_quarter_range() -> None:
    with pytest.raises(ValidationError):
        _make_transcript(quarter=5)
    with pytest.raises(ValidationError):
        _make_transcript(quarter=0)


def test_transcript_validates_ticker_pattern() -> None:
    with pytest.raises(ValidationError):
        _make_transcript(ticker="lowercase")
    with pytest.raises(ValidationError):
        _make_transcript(ticker="")


def test_transcript_allows_empty_qa() -> None:
    """A call where we couldn't detect the Q&A boundary is still a
    valid transcript — the entire body lands in prepared_remarks."""
    t = _make_transcript(q_and_a="")
    assert t.q_and_a == ""


def test_transcript_config_error_is_runtime_error() -> None:
    """RuntimeError subclass — same shape as EdgarConfigError."""
    err = TranscriptConfigError("missing OPENAI_API_KEY")
    assert isinstance(err, RuntimeError)


def test_audio_source_is_runtime_checkable() -> None:
    """The Protocol is `@runtime_checkable` so registry validation
    can confirm a constructed source satisfies the interface."""

    class _StubSource:
        name = "stub"

        def find_audio_url(self, ticker: str, year: int, quarter: int) -> str | None:
            return None

        def download(self, audio_url: str) -> bytes:
            return b""

    assert isinstance(_StubSource(), AudioSource)
