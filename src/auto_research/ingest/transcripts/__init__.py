"""Earnings-call transcript ingest.

Public surface:

    from auto_research.ingest.transcripts import fetch_transcript, Transcript

    t = fetch_transcript(
        ticker="NVDA",
        year=2024,
        quarter=2,
        raw_root=Path("data/raw"),
        manifest_path=Path("data/manifest.parquet"),
    )
    # t is Transcript on a fresh fetch, None on no_coverage or cache hit.

The orchestrator owns the integration of the three independent
pieces: source discovery (`AudioSource.find_audio_url`), audio
download + atomic on-disk persistence (`AudioSource.download` →
`_http.atomic_write_bytes`), and transcription (`WhisperEngine.transcribe`).
It also owns idempotency via the shared `manifest` ledger.

A ticker without a registry entry → status="no_coverage" manifest
row, no fetch attempted, return None. A source that returns None
from `find_audio_url` → same. The orchestrator never silently
retries no_coverage rows (per spec §6.1 "not retried into degraded
data" — the `existing_doc_ids` filter on status=("ok","no_coverage")
treats both as terminal).

See `docs/decisions/2026-05-23-transcripts-source.md` for the source
selection rationale + the deferred-FMP decision.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from auto_research.ingest import _http, manifest
from auto_research.ingest.transcripts import registry
from auto_research.ingest.transcripts._base import (
    AudioSource,
    Transcriber,
    Transcript,
    TranscriptConfigError,
)
from auto_research.ingest.transcripts._whisper import WhisperEngine

SOURCE: Final = "transcripts"

# Statuses we treat as cached (don't re-fetch). `no_coverage` is
# permanent: the source said "no transcript for this quarter" and we
# don't want to keep paying SEC / Whisper / IR-host for the same
# negative answer.
_CACHED_STATUSES: Final = ("ok", "no_coverage")

_logger = logging.getLogger(__name__)

# Run the registry sanity-check at import time so a typo in
# REGISTRY surfaces at startup, not at fetch.
registry.validate()


def _doc_id(ticker: str, year: int, quarter: int) -> str:
    """Manifest dedup key — stable across (ticker, year, quarter)."""
    return f"{ticker.upper()}-{year}Q{quarter}"


def _destination_path(
    ticker: str,
    year: int,
    quarter: int,
    raw_root: Path,
    audio_url: str,
) -> Path:
    """Where the audio bytes land on disk.

    `data/raw/transcripts/{ticker}/{year}/{ticker}-{year}Q{quarter}{ext}`
    where `ext` is derived from the audio URL (defaults to `.bin` if
    none — same fallback shape as the EDGAR path).
    """
    ext = Path(audio_url.split("?", 1)[0]).suffix or ".bin"
    return raw_root / SOURCE / ticker.upper() / str(year) / f"{_doc_id(ticker, year, quarter)}{ext}"


def _open_source(source_name: str) -> AudioSource:
    """Construct the right `AudioSource` for a registry-resolved name.

    Keeps the orchestrator from importing every source unconditionally
    — sources are imported lazily so installing-without-some-source
    (e.g., q4inc with Playwright) still works for the others.
    """
    if source_name == "direct_mp3":
        from auto_research.ingest.transcripts.sources.direct_mp3 import DirectMp3Source

        return DirectMp3Source()
    raise TranscriptConfigError(
        f"Unknown source name {source_name!r}. Registered names: "
        f"{sorted(registry.KNOWN_SOURCES)}."
    )


def _no_coverage_row(
    ticker: str,
    year: int,
    quarter: int,
    source_name: str,
) -> dict[str, object]:
    """Manifest row recording that we tried and the source said no."""
    return {
        "source": SOURCE,
        "entity_id": ticker.upper(),
        "doc_id": _doc_id(ticker, year, quarter),
        "form_type": f"transcript:{source_name}",
        "event_datetime": None,  # no PIT stamp for a non-existent transcript
        "fetched_at": datetime.now(UTC),
        "content_sha256": None,
        "path": None,
        "status": "no_coverage",
    }


def _ok_row(
    transcript: Transcript,
    source_name: str,
    path: Path,
    sha: str,
) -> dict[str, object]:
    return {
        "source": SOURCE,
        "entity_id": transcript.ticker,
        "doc_id": _doc_id(transcript.ticker, transcript.year, transcript.quarter),
        "form_type": f"transcript:{source_name}",
        "event_datetime": transcript.event_datetime,
        "fetched_at": datetime.now(UTC),
        "content_sha256": sha,
        "path": str(path),
        "status": "ok",
    }


def fetch_transcript(
    ticker: str,
    year: int,
    quarter: int,
    *,
    raw_root: Path,
    manifest_path: Path,
    engine: Transcriber | None = None,
    source: AudioSource | None = None,
    event_datetime: datetime | None = None,
) -> Transcript | None:
    """Fetch and transcribe one earnings call. Idempotent via the manifest.

    Idempotency: if `(transcripts, doc_id)` already has a row with
    `status ∈ ("ok", "no_coverage")`, returns None without
    touching the source or Whisper.

    `source` and `engine` are dependency-injection seams for testing
    — production callers leave them None and the orchestrator
    constructs them from the registry / env.

    `event_datetime` is the call start time the caller knows (e.g.,
    from an 8-K Item 2.02 announcement). If None, falls back to
    `datetime.now(UTC)` — acceptable for backfill but Issue #6's AC
    expects callers to supply the real call time when known.
    """
    ticker_upper = ticker.upper()
    doc_id = _doc_id(ticker_upper, year, quarter)

    # Idempotency check first — cheapest path.
    if manifest.contains(
        manifest_path, source=SOURCE, doc_id=doc_id, status=_CACHED_STATUSES
    ):
        _logger.debug("transcript cached: %s", doc_id)
        return None

    # Resolve the source. If the registry has no entry, treat as
    # permanent no_coverage and record it.
    source_name = registry.lookup(ticker_upper)
    if source_name is None:
        _logger.info("no transcript source registered for %s", ticker_upper)
        manifest.append(
            manifest_path,
            [_no_coverage_row(ticker_upper, year, quarter, source_name="unregistered")],
        )
        return None

    owns_source = source is None
    if source is None:
        source = _open_source(source_name)
    owns_engine = engine is None
    if engine is None:
        engine = WhisperEngine()

    try:
        audio_url = source.find_audio_url(ticker_upper, year, quarter)
        if audio_url is None:
            _logger.info(
                "source %s reports no coverage for %s %sQ%s",
                source_name,
                ticker_upper,
                year,
                quarter,
            )
            manifest.append(
                manifest_path,
                [_no_coverage_row(ticker_upper, year, quarter, source_name=source_name)],
            )
            return None

        # Download, then persist atomically (same discipline as EDGAR
        # raw filings — see `_http.atomic_write_bytes`).
        content = source.download(audio_url)
        if not content:
            # Empty body slipped past whatever retry layer the source
            # uses; treat as no_coverage rather than recording a
            # zero-byte file.
            _logger.warning(
                "source %s returned empty body for %s %sQ%s; recording no_coverage",
                source_name,
                ticker_upper,
                year,
                quarter,
            )
            manifest.append(
                manifest_path,
                [_no_coverage_row(ticker_upper, year, quarter, source_name=source_name)],
            )
            return None

        sha = hashlib.sha256(content).hexdigest()
        dest = _destination_path(ticker_upper, year, quarter, raw_root, audio_url)
        _http.atomic_write_bytes(dest, content)

        # Transcribe via Whisper. event_datetime tagged on the result.
        transcript = engine.transcribe(
            content,
            ticker=ticker_upper,
            year=year,
            quarter=quarter,
            event_datetime=event_datetime or datetime.now(UTC),
        )

        manifest.append(manifest_path, [_ok_row(transcript, source_name, dest, sha)])
        return transcript
    finally:
        if owns_engine:
            engine.close()
        if owns_source:
            close_fn = getattr(source, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = [
    "SOURCE",
    "Transcriber",
    "Transcript",
    "TranscriptConfigError",
    "WhisperEngine",
    "fetch_transcript",
]
