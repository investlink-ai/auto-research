"""Per-source audio-discovery + transcript types.

Each issuer of earnings call audio exposes the recording differently
— some serve a direct MP3 on their IR page, others use Q4 Inc's
webcast platform (HLS stream behind JS), some use YouTube replays.
The `AudioSource` Protocol is the seam that lets each platform have
its own extractor while the orchestrator (`fetch_transcript` in
`__init__.py`) stays source-agnostic.

`Transcript` is the shape we return upstream: `ticker`, `year`,
`quarter`, `event_datetime`, `prepared_remarks`, `q_and_a`. Frozen
via `model_config = ConfigDict(frozen=True)` for the same reasons as
universe entries — a transcript is a fact at a point in time and
downstream code mutating it would create silent bugs.

V1 has no speaker diarization. The Q&A boundary is detected by string
match on conventional markers ("Question-and-Answer Session", "We
will now begin the Q&A") in the Whisper output. If signal A2's IC
demands speaker labels, a v2 PR plugs in `pyannote.audio` between the
Whisper transcript and the Q&A split.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptConfigError(RuntimeError):
    """Required env var or config is missing.

    Raised at client construction so failures surface at startup
    rather than during the first fetch. Mirrors the EDGAR pattern
    (`SEC_USER_AGENT` missing → fail loud) for `OPENAI_API_KEY`.
    """


class Transcript(BaseModel):
    """One earnings-call transcript. Frozen — mutation raises ValidationError.

    `event_datetime` is the call start (the issuer's stated start time
    if available, else the audio's publication timestamp). It plays
    the same role as EDGAR's `accepted_datetime`: the canonical
    point-in-time stamp from which Feast's lag-1 cutoff is derived
    downstream (INV-1).

    `prepared_remarks` and `q_and_a` together cover the full call.
    Either may be empty string (a partial call, or a transcript where
    we couldn't detect a Q&A boundary marker — in which case the
    entire transcript text lands in `prepared_remarks`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=8, pattern=r"^[A-Z][A-Z0-9.\-]*$")
    year: int = Field(ge=1900, le=2100)
    quarter: int = Field(ge=1, le=4)
    event_datetime: datetime
    prepared_remarks: str
    q_and_a: str

    @field_validator("event_datetime")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        # INV-1 defense in depth: a naive datetime here would silently
        # land in `manifest.event_datetime` (pa.timestamp(tz='UTC')) and
        # corrupt Feast's lag-1 cutoff downstream. Fail at construction.
        if value.tzinfo is None:
            raise ValueError(
                "event_datetime must be timezone-aware (tzinfo is None). "
                "Pass a UTC datetime (e.g. datetime(..., tzinfo=UTC))."
            )
        return value


@runtime_checkable
class AudioSource(Protocol):
    """A platform-specific audio extractor.

    Each implementation under `transcripts.sources.*` honors this
    contract; the registry maps each universe ticker to one source.
    `find_audio_url` is split from `download` so callers can probe
    coverage cheaply (just the URL discovery, no MB transferred)
    before committing to the download cost.
    """

    name: str  # short identifier — appears in manifest rows and logs

    def find_audio_url(self, ticker: str, year: int, quarter: int) -> str | None:
        """Return the audio URL for one quarter, or None if no coverage.

        None signals 'this source has no transcript for this (ticker,
        year, quarter)' — the orchestrator writes a `status="no_coverage"`
        manifest row and the call returns None. Sources MUST NOT raise
        on missing-coverage; raise only on infrastructure failures
        (network errors, IR-page schema changes, etc).
        """
        ...

    def download(self, audio_url: str) -> bytes:
        """Fetch the audio bytes at `audio_url`.

        Caller is responsible for atomic on-disk persistence via
        `_http.atomic_write_bytes`. This method returns raw bytes so
        sources can use whatever HTTP / streaming / decoding mechanism
        their platform requires (direct GET, HLS reassembly via
        ffmpeg, yt-dlp shell-out, etc.) without leaking those details
        into the orchestrator.
        """
        ...


@runtime_checkable
class Transcriber(Protocol):
    """An audio-to-text engine.

    Symmetric with `AudioSource`: just as each platform plugs in via
    a source, each transcription backend (OpenAI Whisper, AssemblyAI,
    local whisper.cpp, …) plugs in via this Protocol. The orchestrator
    holds a `Transcriber`, never a concrete `WhisperEngine`, so tests
    can substitute fakes and a future engine swap doesn't ripple
    through the orchestrator.
    """

    def transcribe(
        self,
        audio: bytes,
        *,
        ticker: str,
        year: int,
        quarter: int,
        event_datetime: datetime,
    ) -> Transcript:
        """Convert audio bytes into a `Transcript`."""
        ...

    def close(self) -> None:
        """Release any underlying client / connection pool."""
        ...


__all__ = ["AudioSource", "Transcriber", "Transcript", "TranscriptConfigError"]
