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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

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


def _open_youtube() -> AudioSource:
    from auto_research.ingest.transcripts.sources.youtube import YouTubeSource

    return YouTubeSource()


# Source-name → factory. Each factory does its own lazy import so
# unrelated sources (e.g. youtube → yt-dlp) don't load until used.
# The orchestrator asserts `_SOURCE_FACTORIES.keys() == KNOWN_SOURCES`
# at import — a new entry in either set without the other fails loud
# at startup.
_SOURCE_FACTORIES: Final[dict[str, Callable[[], AudioSource]]] = {
    "direct_mp3": _open_direct_mp3,
    "youtube": _open_youtube,
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
    none). Uses `urllib.parse.urlparse` so query strings, fragments,
    `;jsessionid=` path parameters, and userinfo are all stripped in
    one canonical pass before suffix extraction.
    """
    parsed = urlparse(audio_url.strip())
    # urlparse's `path` strips query and fragment but keeps params
    # (the `;`-delimited segment per RFC 3986). Split on `;` to drop
    # `;jsessionid=…`-style suffixes that some IR pages still carry.
    path_only = parsed.path.split(";", 1)[0]
    ext = Path(path_only).suffix or ".bin"
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


_ERROR_DEDUP_KEYS: Final = ("source", "doc_id", "status", "form_type")


def fetch_transcript(
    ticker: str,
    year: int,
    quarter: int,
    *,
    raw_root: Path,
    manifest_path: Path,
    event_datetime: datetime,
    engine: Transcriber | None = None,
    source: AudioSource | None = None,
) -> Transcript | None:
    """Fetch and transcribe one earnings call. Idempotent via the manifest.

    Idempotency: if `(transcripts, doc_id)` already has a row with
    `status ∈ ("ok", "no_coverage")`, returns None without touching
    the source or Whisper. `status="error"` rows do NOT cache — a
    follow-up run retries (registry population, transient empty
    bodies, etc.). Error rows dedup by `(source, doc_id, status,
    form_type)` so distinct failure modes (`transcript:unregistered`
    vs. `transcript:direct_mp3`) coexist rather than colliding.

    `source` and `engine` are dependency-injection seams for testing
    — production callers leave them None and the orchestrator
    constructs them from the registry / env.

    `event_datetime` is the call start time (e.g., from an 8-K Item
    2.02 announcement). REQUIRED: a wall-clock fallback would
    silently violate INV-1 by stamping backfilled transcripts with
    the run time. `Transcript`'s field validator additionally rejects
    naive datetimes (defense in depth).
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
    # cache lock. When the registry is later populated for this
    # ticker, the next call short-circuits past this branch and
    # fetches normally.
    source_name = registry.lookup(ticker_upper)
    if source_name is None:
        _logger.info("no transcript source registered for %s", ticker_upper)
        manifest.append(
            manifest_path,
            [_error_row(ticker_upper, year, quarter, source_name=_UNREGISTERED_SUFFIX)],
            unique_keys=_ERROR_DEDUP_KEYS,
        )
        return None

    owns_source = source is None
    if source is None:
        source = _open_source(source_name)
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

        # Engine constructed here — AFTER find_audio_url so a coverage-
        # survey loop dominated by None responses never pays the
        # OPENAI_API_KEY / ffmpeg cost, but BEFORE source.download so a
        # missing env doesn't leave an orphan audio file on disk.
        owns_engine = engine is None
        if engine is None:
            engine = WhisperEngine()
        try:
            content = source.download(audio_url)
            if not content:
                # Empty body slipped past the source's own retries.
                # Retryable: don't poison the cache with a permanent
                # no_coverage on a transient host hiccup.
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
                    unique_keys=_ERROR_DEDUP_KEYS,
                )
                return None

            sha = hashlib.sha256(content).hexdigest()
            dest = _destination_path(ticker_upper, year, quarter, raw_root, audio_url)
            _http.atomic_write_bytes(dest, content)
            try:
                transcript = engine.transcribe(
                    content,
                    ticker=ticker_upper,
                    year=year,
                    quarter=quarter,
                    event_datetime=event_datetime,
                )
            except Exception:
                # Audio is on disk but transcription failed. Record
                # an error row so the operator can see the orphan
                # has a tracked failure (otherwise the file looks
                # untracked); the row is NOT cached so the next run
                # retries and overwrites.
                _logger.warning(
                    "Whisper transcribe failed for %s %sQ%s — audio at %s, "
                    "recording error row",
                    ticker_upper,
                    year,
                    quarter,
                    dest,
                )
                manifest.append(
                    manifest_path,
                    [_error_row(ticker_upper, year, quarter, source_name=source_name)],
                    unique_keys=_ERROR_DEDUP_KEYS,
                )
                raise
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
