"""OpenAI Whisper API client for earnings-call audio.

Single responsibility: take raw audio bytes (any container ffmpeg
can read — mp3, m4a, mp4, wav, webm) and return a structured
`Transcript`. Source discovery / download is the caller's problem;
see `transcripts.sources.*` and the orchestrator in
`transcripts/__init__.py`.

Why OpenAI Whisper API (`whisper-1`) rather than local
`whisper.cpp`:
- The cost is negligible at our scale (~$0.36 / 60-min call, ~$10/mo
  ongoing for ~90 calls/quarter).
- Quality is identical (the API serves the same `large-v3` weights).
- One less local dependency (whisper.cpp + a model file ~3 GB).
- The `make live-smoke` harness can drive the API; no GPU required.

The 25 MB upload-size limit on the Whisper API is enforced by
chunking the input audio with `ffmpeg` (~10-minute segments at
default 60 kbps voice quality, which keeps each chunk under 5 MB).
Transcripts of each chunk are concatenated.

Q&A boundary detection runs on the concatenated text. Conventional
analyst-call markers ("Question-and-Answer Session", "We will now
begin the Q&A", "I'll turn it over to questions") split the text
into prepared remarks and Q&A. If no marker is found, the whole
transcript lands in `prepared_remarks` and `q_and_a` is the empty
string — downstream signal extractors that care about the Q&A
section MUST handle that case.

`OPENAI_API_KEY` must be set in the environment (read at
construction). Raises `TranscriptConfigError` on missing/blank key
so the failure surfaces at startup rather than mid-fetch.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Final

from openai import OpenAI

from ._base import Transcript, TranscriptConfigError

# Retries are delegated to the OpenAI SDK (`OpenAI(max_retries=N)`),
# which covers 408 / 409 / 429 / 5xx and connection-level errors with
# exponential backoff + Retry-After honoring. No need for a custom
# tenacity layer here — it would duplicate the SDK and obscure the
# actual error surface in tracebacks.
_DEFAULT_MAX_RETRIES: Final = 5

# Whisper API caps single uploads at 25 MB. Stay well under that to
# absorb header overhead — 20 MB target.
_MAX_CHUNK_BYTES: Final = 20 * 1024 * 1024
# Default ffmpeg chunk duration. Earnings calls run ~50-70 min; 10 min
# chunks → 5-7 segments per call, each safely under the 20 MB cap at
# 64 kbps voice quality.
_CHUNK_SECONDS: Final = 600
# Floor on chunk duration. Below this, a 70-min call would explode
# into hundreds of Whisper calls (each $0.006). Caller can override
# only above this floor to keep API spend bounded.
_MIN_CHUNK_SECONDS: Final = 30
# Hard ceiling on the per-call ffmpeg subprocess. A 70-min audio at
# realtime decode is ~70 min; we give ffmpeg 30 min and call it stuck.
_DEFAULT_FFMPEG_TIMEOUT: Final = 1800.0

# Q&A boundary markers — ordered most-specific to most-generic. The
# first match wins; subsequent markers are ignored. Real earnings
# calls almost always have at least one of these phrasings.
#
# Pattern 4 ("we'll/let's take/open ... questions") is the most
# permissive marker and would otherwise false-fire on mid-prepared-
# remarks asides like "we'll take questions in a moment". The
# `^\s*(?:Operator:\s*)?` anchor (with re.MULTILINE) requires the
# phrase to START a line — the convention for the operator handoff
# in real earnings transcripts.
_QA_BOUNDARY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bQuestion[-\s]and[-\s]Answer\s+Session\b", re.IGNORECASE),
    re.compile(r"\bWe(?:'ll| will)\s+now\s+(?:begin|move)\s+(?:the\s+)?Q\s*&\s*A\b", re.IGNORECASE),
    re.compile(r"\bI(?:'ll| will)\s+(?:now\s+)?turn\s+(?:it|the call)\s+over\s+to\s+questions\b", re.IGNORECASE),
    re.compile(
        r"^\s*(?:Operator:\s*)?(?:we'll|we will|let's)\s+(?:now\s+)?(?:take|open).*?\bquestions\b",
        re.IGNORECASE | re.MULTILINE,
    ),
)


def _resolve_api_key(api_key: str | None) -> str:
    raw = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
    cleaned = raw.strip()
    if not cleaned:
        raise TranscriptConfigError(
            "OpenAI Whisper API requires OPENAI_API_KEY. Set it in the "
            "environment, or pass `api_key=` to WhisperEngine explicitly."
        )
    return cleaned


def _ensure_ffmpeg() -> str:
    """Locate ffmpeg or raise. ffmpeg-python could replace this, but
    a `subprocess` call to the system binary is one less dependency
    and avoids the per-process startup cost of a Python wrapper.
    """
    path = shutil.which("ffmpeg")
    if not path:
        raise TranscriptConfigError(
            "ffmpeg not found on PATH. Whisper audio chunking requires "
            "ffmpeg. On macOS: `brew install ffmpeg`. On Debian/Ubuntu: "
            "`apt-get install ffmpeg`."
        )
    return path


def _split_qa(full_text: str) -> tuple[str, str]:
    """Detect the prepared-remarks → Q&A boundary in `full_text`.

    Returns `(prepared_remarks, q_and_a)`. If no marker is found,
    `(full_text, "")` — the whole transcript lands in prepared remarks.
    Downstream Q&A-specific signals must check for an empty q_and_a.
    """
    for pattern in _QA_BOUNDARY_PATTERNS:
        match = pattern.search(full_text)
        if match:
            return full_text[: match.start()].rstrip(), full_text[match.start() :].lstrip()
    return full_text, ""


class WhisperEngine:
    """Thin OpenAI Whisper API wrapper.

    Constructed once per ingest run; reused across calls. The
    underlying `openai.OpenAI` client is opened eagerly in `__init__`.
    Always close it — either via `.close()` or `with` form.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "whisper-1",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        chunk_seconds: int = _CHUNK_SECONDS,
        ffmpeg_path: str | None = None,
        ffmpeg_timeout: float = _DEFAULT_FFMPEG_TIMEOUT,
    ) -> None:
        if chunk_seconds < _MIN_CHUNK_SECONDS:
            raise ValueError(
                f"chunk_seconds={chunk_seconds} is below the {_MIN_CHUNK_SECONDS}s "
                f"floor; smaller values risk uncapped Whisper API spend "
                f"(each chunk is a separate billed call)."
            )
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if ffmpeg_timeout <= 0:
            raise ValueError(f"ffmpeg_timeout must be positive, got {ffmpeg_timeout}")
        # Resolve cheap (env / PATH) dependencies BEFORE opening the
        # OpenAI client. Otherwise a missing ffmpeg would leak an
        # already-constructed httpx connection pool inside `OpenAI(...)`.
        resolved_api_key = _resolve_api_key(api_key)
        self._ffmpeg = ffmpeg_path or _ensure_ffmpeg()
        # SDK handles retries for 408 / 409 / 429 / 5xx + connection
        # errors, with exponential backoff and Retry-After honoring.
        self._client = OpenAI(api_key=resolved_api_key, max_retries=max_retries)
        self._model = model
        self._chunk_seconds = chunk_seconds
        self._ffmpeg_timeout = ffmpeg_timeout

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> WhisperEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def transcribe(
        self,
        audio: bytes,
        *,
        ticker: str,
        year: int,
        quarter: int,
        event_datetime: datetime,
    ) -> Transcript:
        """Transcribe one earnings call's audio bytes.

        Strategy:
        1. Write `audio` to a tmp file (ffmpeg needs a seekable input).
        2. Probe duration via ffmpeg; if under chunk_seconds, send in
           one shot. Otherwise split into chunk-sized segments.
        3. Call Whisper API per chunk; concatenate the text.
        4. Apply Q&A boundary detection.

        Per-chunk retries (408/409/429/5xx + connection errors) are
        handled by the OpenAI SDK's `max_retries` setting on the
        client.
        """
        with tempfile.TemporaryDirectory(prefix="whisper-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_path = tmpdir_path / "source.audio"
            source_path.write_bytes(audio)
            segments = self._chunk_audio(source_path, tmpdir_path)
            full_text_parts: list[str] = []
            for segment in segments:
                full_text_parts.append(self._transcribe_one(segment))
        full_text = "\n".join(part.strip() for part in full_text_parts if part.strip())
        prepared, qa = _split_qa(full_text)
        return Transcript(
            ticker=ticker,
            year=year,
            quarter=quarter,
            event_datetime=event_datetime,
            prepared_remarks=prepared,
            q_and_a=qa,
        )

    def _chunk_audio(self, source: Path, workdir: Path) -> list[Path]:
        """Split `source` into ~`chunk_seconds`-long segments via ffmpeg.

        Single-pass `-f segment` invocation; ffmpeg writes each output
        chunk as `chunk-000.mp3`, `chunk-001.mp3`, … The encode is mp3
        at 64 kbps (voice-quality, well under Whisper's 25 MB cap for
        even 30+ minute chunks). If the source is shorter than the
        chunk length, ffmpeg emits a single output file.
        """
        out_pattern = workdir / "chunk-%03d.mp3"
        cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-f",
            "segment",
            "-segment_time",
            str(self._chunk_seconds),
            "-acodec",
            "libmp3lame",
            "-b:a",
            "64k",
            "-vn",
            str(out_pattern),
        ]
        # `timeout=` guards against ffmpeg hanging on malformed input
        # (corrupt mp4 atoms, stalled libavformat reads). Without it a
        # single bad file wedges the worker and blocks the manifest
        # lock for every other writer on the host.
        try:
            subprocess.run(cmd, check=True, timeout=self._ffmpeg_timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffmpeg exceeded {self._ffmpeg_timeout:.0f}s timeout on {source}. "
                "Likely corrupt or unrecognized audio container."
            ) from exc
        chunks = sorted(workdir.glob("chunk-*.mp3"))
        if not chunks:
            raise RuntimeError(
                f"ffmpeg produced no chunks from {source}. "
                "Possible cause: source file is not decodable as audio."
            )
        for chunk in chunks:
            size = chunk.stat().st_size
            if size > _MAX_CHUNK_BYTES:
                raise RuntimeError(
                    f"Chunk {chunk.name} is {size} bytes (>{_MAX_CHUNK_BYTES} "
                    f"limit). Reduce chunk_seconds or lower the bitrate."
                )
        return chunks

    def _transcribe_one(self, chunk: Path) -> str:
        """Send one chunk to the OpenAI Whisper API and return its text.

        The SDK handles retries for 408 / 409 / 429 / 5xx + connection
        errors (configured via `max_retries` on the client). Errors
        that escape this call are post-retry-budget failures.
        """
        with chunk.open("rb") as fh:
            resp = self._client.audio.transcriptions.create(
                model=self._model,
                file=fh,
                response_format="text",
            )
        # response_format="text" returns a plain `str` on openai>=2.38.
        # If a future SDK swaps that for a typed wrapper, `str(resp)`
        # would silently produce a `Transcription(text=...)` repr and
        # corrupt every transcript. Assert the shape we depend on.
        if isinstance(resp, str):
            return resp.strip()
        text = getattr(resp, "text", None)
        if isinstance(text, str):
            return text.strip()
        raise TypeError(
            f"OpenAI Whisper returned {type(resp).__name__}; expected "
            "`str` (response_format='text') or an object with a `.text` "
            "attribute. SDK contract drift — pin openai version."
        )


__all__ = ["WhisperEngine"]
