"""Unit tests for `auto_research.ingest.transcripts._whisper`.

The OpenAI client is monkeypatched — we never hit the real Whisper
API. ffmpeg IS invoked against a small synthetic WAV fixture to
exercise the chunking path; ffmpeg presence is assumed (the repo's
CI workflow installs it explicitly for these tests).

Tests cover:
- `_resolve_api_key` validation (missing / blank → `TranscriptConfigError`)
- `_ensure_ffmpeg` validation
- `_split_qa` boundary detection across the documented marker variants
- chunking via ffmpeg yields ≥1 segment for any decodable audio
- the full `transcribe` path with a fake OpenAI client
"""

from __future__ import annotations

import os
import shutil
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from auto_research.ingest.transcripts._base import Transcript, TranscriptConfigError
from auto_research.ingest.transcripts._whisper import (
    WhisperEngine,
    _ensure_ffmpeg,
    _resolve_api_key,
    _split_qa,
)


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_silent_wav(seconds: float = 1.0, sample_rate: int = 8000) -> bytes:
    """Synthesize a minimal silent WAV in-memory. Tiny (~16 KB/sec)."""
    n_samples = int(seconds * sample_rate)
    pcm = b"\x00\x00" * n_samples  # 16-bit mono silence
    data_size = len(pcm)
    chunk_size = 36 + data_size
    header = b"RIFF" + struct.pack("<I", chunk_size) + b"WAVE"
    fmt = (
        b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", 1)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", sample_rate * 2)
        + struct.pack("<H", 2)
        + struct.pack("<H", 16)
    )
    data = b"data" + struct.pack("<I", data_size) + pcm
    return header + fmt + data


# ---------- _resolve_api_key ----------


def test_resolve_api_key_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert _resolve_api_key(None) == "sk-test"


def test_resolve_api_key_prefers_explicit_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert _resolve_api_key("explicit-key") == "explicit-key"


def test_resolve_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(TranscriptConfigError, match="OPENAI_API_KEY"):
        _resolve_api_key(None)


def test_resolve_api_key_treats_blank_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(TranscriptConfigError):
        _resolve_api_key(None)


# ---------- _ensure_ffmpeg ----------


def test_ensure_ffmpeg_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(TranscriptConfigError, match="ffmpeg"):
        _ensure_ffmpeg()


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_ensure_ffmpeg_returns_path_when_present() -> None:
    path = _ensure_ffmpeg()
    assert os.path.isfile(path) or os.access(path, os.X_OK)


# ---------- _split_qa ----------


def test_split_qa_recognizes_canonical_marker() -> None:
    text = "Prepared remarks here.\n\nQuestion-and-Answer Session\n\nOperator: First question…"
    prepared, qa = _split_qa(text)
    assert prepared == "Prepared remarks here."
    assert qa.startswith("Question-and-Answer Session")


def test_split_qa_recognizes_we_will_now_begin() -> None:
    text = "Thanks for joining.\n\nWe will now begin the Q&A.\n\nOperator: …"
    prepared, qa = _split_qa(text)
    assert prepared == "Thanks for joining."
    assert qa.startswith("We will now begin")


def test_split_qa_recognizes_turn_over_to_questions() -> None:
    text = "…and that concludes my remarks. I'll now turn it over to questions.\nOperator: Thanks."
    prepared, qa = _split_qa(text)
    assert prepared.endswith("my remarks.")
    assert "turn it over to questions" in qa


def test_split_qa_returns_full_text_when_no_marker() -> None:
    text = "Just prepared remarks, no Q&A boundary detected."
    prepared, qa = _split_qa(text)
    assert prepared == text
    assert qa == ""


def test_split_qa_first_match_wins() -> None:
    """If two markers appear, the earlier one defines the boundary."""
    text = "Remarks.\nQuestion-and-Answer Session\nOperator: We will now begin the Q&A."
    prepared, qa = _split_qa(text)
    assert prepared == "Remarks."
    assert qa.count("Question-and-Answer Session") == 1


# ---------- chunking + transcribe (with fake OpenAI client) ----------


class _FakeTranscriptionsClient:
    def __init__(self, *, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[Path] = []

    def create(self, *, model: str, file: Any, response_format: str) -> str:
        assert model == "whisper-1"
        assert response_format == "text"
        # `file` is an open file handle — record its path for the test.
        self.calls.append(Path(file.name))
        return self._response_text


class _FakeAudio:
    def __init__(self, transcriptions: _FakeTranscriptionsClient) -> None:
        self.transcriptions = transcriptions


class _FakeOpenAI:
    def __init__(self, transcriptions: _FakeTranscriptionsClient) -> None:
        self.audio = _FakeAudio(transcriptions)

    def close(self) -> None:
        pass


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_whisper_engine_transcribe_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end through a fake OpenAI client; ffmpeg chunks real audio."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    transcriptions = _FakeTranscriptionsClient(
        response_text=(
            "Thanks for joining today's Q2 earnings call. Revenue was $30B.\n"
            "Question-and-Answer Session\n"
            "Operator: First question."
        ),
    )
    monkeypatch.setattr(
        "auto_research.ingest.transcripts._whisper.OpenAI",
        lambda *, api_key, max_retries=2: _FakeOpenAI(transcriptions),
    )
    engine = WhisperEngine()
    transcript = engine.transcribe(
        _make_silent_wav(seconds=2.0),
        ticker="NVDA",
        year=2024,
        quarter=2,
        event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
    )
    assert isinstance(transcript, Transcript)
    assert transcript.ticker == "NVDA"
    assert "Revenue was $30B" in transcript.prepared_remarks
    assert "First question" in transcript.q_and_a
    # ffmpeg produced at least one chunk and Whisper was called.
    assert len(transcriptions.calls) >= 1


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_whisper_engine_concatenates_multiple_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-enough audio gets split, and chunk transcripts are joined."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    transcriptions = _FakeTranscriptionsClient(response_text="segment text")
    monkeypatch.setattr(
        "auto_research.ingest.transcripts._whisper.OpenAI",
        lambda *, api_key, max_retries=2: _FakeOpenAI(transcriptions),
    )
    # 90s audio, 30s chunks → 3 segments. (chunk_seconds floor is 30s
    # to bound Whisper API spend; we go just above the floor.)
    engine = WhisperEngine(chunk_seconds=30)
    transcript = engine.transcribe(
        _make_silent_wav(seconds=90.0),
        ticker="NVDA",
        year=2024,
        quarter=2,
        event_datetime=datetime(2024, 5, 22, tzinfo=UTC),
    )
    assert len(transcriptions.calls) >= 2, "expected multiple chunks"
    # Each chunk contributes "segment text"; the join uses newlines.
    assert transcript.prepared_remarks.count("segment text") == len(transcriptions.calls)


def test_whisper_engine_requires_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(TranscriptConfigError):
        WhisperEngine()


def test_whisper_engine_rejects_sub_floor_chunk_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the 30s floor, a 70-min call would explode into hundreds
    of $0.006 Whisper calls. Constructor fails loud."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="chunk_seconds"):
        WhisperEngine(chunk_seconds=10)


def test_whisper_engine_rejects_negative_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_retries=0 is valid (no retries); negative is not."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="max_retries"):
        WhisperEngine(max_retries=-1)


def test_whisper_engine_passes_max_retries_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructor must hand max_retries to the OpenAI client —
    that's where retry semantics live now (not in our own loop)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict[str, int] = {}

    class _SpyOpenAI:
        def __init__(self, *, api_key: str, max_retries: int) -> None:
            captured["max_retries"] = max_retries

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "auto_research.ingest.transcripts._whisper.OpenAI",
        _SpyOpenAI,
    )
    if shutil.which("ffmpeg") is None:
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/ffmpeg")
    WhisperEngine(max_retries=7)
    assert captured["max_retries"] == 7


def test_whisper_engine_rejects_non_positive_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="ffmpeg_timeout"):
        WhisperEngine(ffmpeg_timeout=0)


def test_split_qa_pattern_4_requires_line_anchor() -> None:
    """Pattern 4 ('we'll/let's take/open ... questions') must require
    start-of-line — otherwise a mid-prepared-remarks reference like
    'let's take a moment to address recent questions' would false-fire
    and corrupt the split. The fix anchors pattern 4 with `^` +
    re.MULTILINE so only line-starting markers count."""
    text = (
        "Thanks for joining. Before we begin, let's take a moment to "
        "address some recent questions about strategy.\n"
        "Operator: We'll now open the floor for questions.\n"
        "First question please."
    )
    prepared, qa = _split_qa(text)
    # The mid-paragraph "let's take ... questions" must NOT trigger;
    # it stays in prepared remarks.
    assert "let's take a moment" in prepared
    assert "recent questions about strategy" in prepared
    # The line-anchored operator handoff IS the boundary.
    assert qa.startswith("Operator: We'll now open")


# ---------- Construction / resource ordering ----------


def test_whisper_engine_resolves_ffmpeg_before_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ffmpeg is missing, __init__ must fail BEFORE constructing
    the OpenAI client — otherwise an httpx connection pool is opened
    and never closed, leaking on every retry of construction."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(shutil, "which", lambda _: None)  # ffmpeg missing

    constructed: list[bool] = []

    class _SpyOpenAI:
        def __init__(self, *, api_key: str, max_retries: int = 2) -> None:
            constructed.append(True)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "auto_research.ingest.transcripts._whisper.OpenAI",
        _SpyOpenAI,
    )
    with pytest.raises(TranscriptConfigError, match="ffmpeg"):
        WhisperEngine()
    assert constructed == [], (
        "OpenAI client must not be opened before ffmpeg check — "
        "would leak httpx pool on every retry of construction"
    )
