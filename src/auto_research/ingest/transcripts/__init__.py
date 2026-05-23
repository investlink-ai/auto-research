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
import re
from collections.abc import Callable
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
# negative answer. `error` is intentionally NOT here — a transient
# empty body or an unregistered ticker can be retried after the
# underlying state is fixed.
_CACHED_STATUSES: Final = ("ok", "no_coverage")

# Manifest form_type suffix for the "ticker has no registry entry"
# case. Distinct from "registered source said no coverage" so a coverage
# audit can tell the two apart.
_UNREGISTERED_SUFFIX: Final = "unregistered"

_logger = logging.getLogger(__name__)


def _open_direct_mp3() -> AudioSource:
    from auto_research.ingest.transcripts.sources.direct_mp3 import DirectMp3Source

    return DirectMp3Source()


# Source-name → factory. Each factory does its own lazy import so
# unrelated sources (q4inc with Playwright, youtube with yt-dlp) don't
# need to be installed for the others to work. The orchestrator
# asserts `_SOURCE_FACTORIES.keys() == KNOWN_SOURCES` at import — a
# new entry in either set without the other fails loud at startup.
_SOURCE_FACTORIES: Final[dict[str, Callable[[], AudioSource]]] = {
    "direct_mp3": _open_direct_mp3,
}

# Run the registry sanity-check at import time so a typo in REGISTRY
# surfaces at startup, not at fetch.
registry.validate()
# Drift guard: every KNOWN_SOURCES name must have a constructor here.
_missing_factories = registry.KNOWN_SOURCES - _SOURCE_FACTORIES.keys()
_orphan_factories = _SOURCE_FACTORIES.keys() - registry.KNOWN_SOURCES
if _missing_factories or _orphan_factories:
    raise RuntimeError(
        f"Source factory drift: missing factories for {sorted(_missing_factories)}, "
        f"orphan factories for {sorted(_orphan_factories)}. KNOWN_SOURCES and "
        "_SOURCE_FACTORIES must stay in lockstep."
    )


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
    none). Both `?query` and `#fragment` are stripped before suffix
    extraction — a URL like `.../audio.mp3#t=120` would otherwise
    bake `.mp3#t=120` into the filename.
    """
    url_path = re.split(r"[?#]", audio_url, maxsplit=1)[0]
    ext = Path(url_path).suffix or ".bin"
    return raw_root / SOURCE / ticker.upper() / str(year) / f"{_doc_id(ticker, year, quarter)}{ext}"


def _open_source(source_name: str) -> AudioSource:
    """Construct the right `AudioSource` for a registry-resolved name."""
    factory = _SOURCE_FACTORIES.get(source_name)
    if factory is None:
        raise TranscriptConfigError(
            f"Unknown source name {source_name!r}. Registered names: "
            f"{sorted(_SOURCE_FACTORIES)}."
        )
    return factory()


def _no_coverage_row(
    ticker: str,
    year: int,
    quarter: int,
    source_name: str,
) -> dict[str, object]:
    """Manifest row for a permanent negative: source said 'no audio'.

    Cached via `_CACHED_STATUSES` — the past doesn't change, so we
    never retry. Distinct from `_error_row`, which records a transient
    failure that SHOULD be retried.
    """
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


def _error_row(
    ticker: str,
    year: int,
    quarter: int,
    source_name: str,
) -> dict[str, object]:
    """Manifest row for a transient failure: empty body, unregistered ticker, etc.

    `status="error"` is NOT in `_CACHED_STATUSES`, so the next fetch
    retries — important so (a) populating the registry doesn't lock a
    ticker out forever via an earlier "unregistered" row, and (b) a
    one-off empty body doesn't poison the cache.
    """
    return {
        "source": SOURCE,
        "entity_id": ticker.upper(),
        "doc_id": _doc_id(ticker, year, quarter),
        "form_type": f"transcript:{source_name}",
        "event_datetime": None,
        "fetched_at": datetime.now(UTC),
        "content_sha256": None,
        "path": None,
        "status": "error",
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
    `status ∈ ("ok", "no_coverage")`, returns None without touching
    the source or Whisper. `status="error"` rows do NOT cache — a
    follow-up run retries (registry population, transient empty
    bodies, etc.).

    `source` and `engine` are dependency-injection seams for testing
    — production callers leave them None and the orchestrator
    constructs them from the registry / env.

    `event_datetime` is the call start time (e.g., from an 8-K Item
    2.02 announcement). Required for the ok path: a fallback to
    `datetime.now(UTC)` would silently violate INV-1 by stamping
    backfilled transcripts with the run wall-clock time. If omitted
    and a real transcript would be produced, raises
    `TranscriptConfigError`. Tz-aware datetime required; `Transcript`
    enforces this in defense-in-depth.
    """
    ticker_upper = ticker.upper()
    doc_id = _doc_id(ticker_upper, year, quarter)

    # Idempotency check first — cheapest path.
    if manifest.contains(
        manifest_path, source=SOURCE, doc_id=doc_id, status=_CACHED_STATUSES
    ):
        _logger.debug("transcript cached: %s", doc_id)
        return None

    # Unregistered ticker → retryable error row, not a permanent
    # cache lock. When PR #6f populates the registry, the next call
    # short-circuits past this branch and fetches normally.
    source_name = registry.lookup(ticker_upper)
    if source_name is None:
        _logger.info("no transcript source registered for %s", ticker_upper)
        manifest.append(
            manifest_path,
            [_error_row(ticker_upper, year, quarter, source_name=_UNREGISTERED_SUFFIX)],
        )
        return None

    owns_source = source is None
    if source is None:
        source = _open_source(source_name)
    # Engine construction is intentionally deferred until we know we
    # actually need to transcribe — a coverage-survey loop dominated
    # by `find_audio_url is None` shouldn't require OPENAI_API_KEY +
    # ffmpeg to be present.
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
            # uses; record as error (retryable) rather than caching a
            # known-bad result forever.
            _logger.warning(
                "source %s returned empty body for %s %sQ%s; recording error",
                source_name,
                ticker_upper,
                year,
                quarter,
            )
            manifest.append(
                manifest_path,
                [_error_row(ticker_upper, year, quarter, source_name=source_name)],
            )
            return None

        # INV-1 guard: we are about to stamp the manifest's
        # event_datetime column from the caller's value. Refuse the
        # silent now() fallback — a wrong PIT stamp here propagates
        # into Feast's lag-1 cutoff and corrupts every downstream
        # backtest that joins on this transcript.
        if event_datetime is None:
            raise TranscriptConfigError(
                f"event_datetime is required when transcribing "
                f"{ticker_upper} {year}Q{quarter}; a wall-clock fallback "
                "would violate INV-1 (point-in-time discipline)."
            )

        sha = hashlib.sha256(content).hexdigest()
        dest = _destination_path(ticker_upper, year, quarter, raw_root, audio_url)
        _http.atomic_write_bytes(dest, content)

        owns_engine = engine is None
        if engine is None:
            engine = WhisperEngine()
        try:
            transcript = engine.transcribe(
                content,
                ticker=ticker_upper,
                year=year,
                quarter=quarter,
                event_datetime=event_datetime,
            )
            manifest.append(manifest_path, [_ok_row(transcript, source_name, dest, sha)])
            return transcript
        finally:
            if owns_engine:
                engine.close()
    finally:
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
