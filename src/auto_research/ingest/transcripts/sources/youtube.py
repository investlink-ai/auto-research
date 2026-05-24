"""`youtube` source: earnings-call audio via yt-dlp.

Most large- and mid-cap earnings calls are mirrored on YouTube within
hours by aggregator channels (Benzinga, Castify Earnings Call, EARNMOAR,
Investing 101, Yahoo Finance) plus the occasional first-party upload.
A universe-wide probe of the 81-ticker v1 universe found a full-length
match (40-90 min duration, the consistent earnings-call shape) for
every ticker when queried by company name. yt-dlp fetches the audio
stream directly — no headless browser, no platform registration.

`find_audio_url` runs a `ytsearch{N}:{query}` lookup, filters results
by duration band AND by a title check that requires the company,
quarter, and an acceptable year framing all appear in the entry's
title. `download` invokes yt-dlp again to extract the best audio
stream as bytes (m4a/AAC), then validates the returned bytes are
actually audio (size floor + magic-byte sniff).

(year, quarter) convention: `quarter` is 1-4. `year` accepts either
the calendar OR the fiscal-year framing of the call — aggregators
title uploads inconsistently (e.g., NVDA's Q4 FY26 call held Feb 2026
is variously titled 'Q4 2026', 'Q4 FY26', 'Q4 FY2026'), so the title
check tolerates `year`, `year-1`, `FY{year%100}`, and `FY{(year-1)%100}`
to match either convention. The caller chooses one consistently for
`doc_id` purposes (the orchestrator stamps `TICKER-{year}Q{quarter}`).

Search query: by default `{ticker} earnings call`, but per-ticker
overrides in `TICKER_QUERIES` let us supply the company name when
the ticker symbol clashes with unrelated YouTube content (a real
problem for tickers like MIR / LEU / FORM whose tickers collide with
crypto and other topics).

Test seam: `YoutubeDLFactory` returns a `yt_dlp.YoutubeDL`-like object
with `extract_info`. Production uses real yt-dlp; tests inject a fake
that yields canned search results + download bytes.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Final, Protocol

from auto_research.ingest.rate_limit import TokenBucket
from auto_research.ingest.transcripts._base import TranscriptConfigError

SOURCE_NAME: Final = "youtube"

# Per-ticker search query override. The default query is
# `{ticker} earnings call`, which works for most tickers but
# false-positives on short tickers whose symbols collide with crypto
# tokens, file extensions, etc. Override here when the ticker's
# symbol is ambiguous; the value is appended with ` earnings call`.
#
# Adding a ticker = adding a row here. The coverage-survey worker
# is the canonical place to populate this en masse; one entry is
# seeded as a canary for the live path.
TICKER_QUERIES: dict[str, str] = {
    "NVDA": "NVIDIA",
}

# Earnings calls are typically 40-90 minutes (2400-5400 sec). We use a
# slightly wider band (30-100 min) to absorb shorter calls (small caps,
# pre-recorded statements) and longer ones (with extended Q&A).
_MIN_DURATION_SEC: Final = 1800
_MAX_DURATION_SEC: Final = 6000

# Default number of search results to retrieve per query. The
# duration + title filters then pick the first qualifying result. 5
# is enough for the universe: aggregators usually appear in the top 3.
_DEFAULT_SEARCH_LIMIT: Final = 5

# Rate budget for `download` / `find_audio_url` invocations. yt-dlp
# internally makes many HTTP requests per call (manifest + N segments);
# this limiter only paces the outer invocation cadence. The inner
# request volume is controlled by the per-invocation yt-dlp opts
# (`sleep_interval`, `sleep_interval_requests`).
_DEFAULT_RATE_PER_SEC: Final = 2.0

# Per-invocation throttle settings handed to yt-dlp. These shape the
# inner request volume (which our TokenBucket doesn't see) so a
# single download isn't 50+ unthrottled fetches to YouTube CDNs.
_YTDLP_SLEEP_INTERVAL: Final = 0.5  # min seconds between requests
_YTDLP_MAX_SLEEP_INTERVAL: Final = 2.0  # max with jitter

# Audio file content gates. Real earnings-call audio is multi-MB; a
# few-KB payload is almost certainly an error page or partial fetch.
_MIN_AUDIO_BYTES: Final = 100_000  # 100 KB
# Magic-byte signatures for common YouTube audio containers. A file
# whose first 16 bytes don't match any of these almost certainly
# isn't decodable audio — typically yt-dlp wrote an HTML error page
# or JSON challenge response. Reject loud rather than feeding it to
# Whisper (where it would burn API budget on garbage).
_AUDIO_MAGIC_PREFIXES: Final[tuple[bytes, ...]] = (
    b"ID3",  # MP3 with ID3v2 header
    b"OggS",  # Ogg / Opus
    b"RIFF",  # WAV / RIFF container
    b"\x1a\x45\xdf\xa3",  # Matroska / WebM EBML header
    b"fLaC",  # FLAC
)
# MP3 ADTS frames start with sync word 0xFFFx. M4A/MP4 files have
# `ftyp` at byte offset 4. These need offset-aware matching.

_logger = logging.getLogger(__name__)


def _looks_like_audio(head: bytes) -> bool:
    """Return True if `head` (first ~16 bytes of a file) is a plausible
    audio container header. Used to reject HTML error pages, JSON
    challenge responses, and other non-audio payloads yt-dlp can
    occasionally write under anti-bot conditions."""
    if len(head) < 4:
        return False
    if any(head.startswith(prefix) for prefix in _AUDIO_MAGIC_PREFIXES):
        return True
    # MP4 / M4A: `ftyp` at offset 4 (after a 4-byte size).
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return True
    # MP3 ADTS sync word: 0xFFFx in first two bytes.
    return bool(head[0] == 0xFF and (head[1] & 0xF0) == 0xF0)


def _year_tokens(year: int) -> tuple[str, ...]:
    """Acceptable year framings in aggregator titles.

    Aggregators title uploads variously: 'NVIDIA Q4 2026 Earnings
    Call' (calendar), 'NVIDIA Q4 FY26' (fiscal-year shortform),
    'Q4 FY2026' (fiscal-year longform). Some title with the calendar
    year of the prior period (Q4 ending Jan 2026 → 'Q4 2025'). The
    title-match filter accepts any of these framings.
    """
    return (
        str(year),
        str(year - 1),
        f"fy{year % 100:02d}",
        f"fy{(year - 1) % 100:02d}",
        f"fy{year}",
        f"fy{year - 1}",
    )


def _title_matches_call(
    title: str, *, company: str, ticker: str, year: int, quarter: int
) -> bool:
    """Verify an aggregator's video title plausibly refers to the
    requested earnings call.

    Loose-but-targeted: the title must contain (company OR ticker
    OR $ticker), the quarter token (`Q{quarter}`), AND any accepted
    year framing. Without this gate, a duration-band search picks
    the first 30-100-minute video YouTube ranks first — which can
    be a different ticker's call, a different quarter's call, or a
    podcast about the company.
    """
    t = title.lower()
    company_l = company.lower()
    ticker_l = ticker.lower()
    if (
        company_l not in t
        and ticker_l not in t
        and f"${ticker_l}" not in t
    ):
        return False
    if f"q{quarter}" not in t:
        return False
    return any(tok in t for tok in _year_tokens(year))


class _YoutubeDLLike(Protocol):
    """The narrow yt-dlp surface this source uses.

    Real `yt_dlp.YoutubeDL` satisfies this Protocol structurally; tests
    can inject any class that implements both `extract_info` and the
    context-manager methods.
    """

    def extract_info(
        self, url: str, *, download: bool = False
    ) -> dict[str, Any] | None: ...

    def __enter__(self) -> _YoutubeDLLike: ...

    def __exit__(self, *args: object) -> None: ...


# yt-dlp opens in a context-manager idiom (`with YoutubeDL(opts) as ydl:`).
# Type the factory's return as an AbstractContextManager so the
# `with self._factory(...)` site checks cleanly without type-ignores.
YoutubeDLFactory = Callable[[dict[str, Any]], AbstractContextManager[_YoutubeDLLike]]


def _default_factory(opts: dict[str, Any]) -> AbstractContextManager[_YoutubeDLLike]:
    """Production yt-dlp factory. Import probed eagerly at module
    load (via `_verify_yt_dlp_available` in `YouTubeSource.__init__`)
    so a missing install fails loud as `TranscriptConfigError`
    instead of mid-fetch."""
    from yt_dlp import YoutubeDL  # type: ignore[import-untyped]

    client: AbstractContextManager[_YoutubeDLLike] = YoutubeDL(opts)
    return client


def _verify_yt_dlp_available() -> None:
    """Probe `import yt_dlp` at YouTubeSource construction time so a
    missing install raises a typed config error at startup rather
    than surfacing as a no_coverage row mid-fetch."""
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise TranscriptConfigError(
            "yt-dlp is not installed. Run `uv sync` or "
            "`pip install yt-dlp>=2026.3`."
        ) from exc


class YouTubeSource:
    """Earnings audio for any ticker whose call gets mirrored on YouTube.

    `find_audio_url` runs `ytsearch{N}:{query}` and returns the first
    result whose duration falls in the earnings-call band AND whose
    title plausibly matches the requested (ticker, year, quarter).
    `download` invokes yt-dlp's audio extractor and returns the file
    bytes for the orchestrator to persist (after a magic-byte check).
    """

    name = SOURCE_NAME

    def __init__(
        self,
        *,
        rate_limiter: TokenBucket | None = None,
        search_limit: int = _DEFAULT_SEARCH_LIMIT,
        min_duration_sec: int = _MIN_DURATION_SEC,
        max_duration_sec: int = _MAX_DURATION_SEC,
        factory: YoutubeDLFactory = _default_factory,
        verify_yt_dlp: bool = True,
    ) -> None:
        if search_limit < 1:
            raise ValueError(f"search_limit must be >= 1, got {search_limit}")
        if min_duration_sec < 0 or max_duration_sec <= min_duration_sec:
            raise ValueError(
                f"invalid duration band: ({min_duration_sec}, {max_duration_sec})"
            )
        if verify_yt_dlp:
            _verify_yt_dlp_available()
        self.rate_limiter = rate_limiter or TokenBucket(rate=_DEFAULT_RATE_PER_SEC)
        self._search_limit = search_limit
        self._min_duration_sec = min_duration_sec
        self._max_duration_sec = max_duration_sec
        self._factory = factory

    def close(self) -> None:
        # Each yt-dlp invocation creates and destroys its own client;
        # no instance-level resources to release. Method exists for
        # parity with other sources that own a long-lived client.
        return

    def __enter__(self) -> YouTubeSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def find_audio_url(self, ticker: str, year: int, quarter: int) -> str | None:
        """Return a YouTube URL whose duration AND title plausibly match
        the requested earnings call.

        Returns None if no result in the duration band passes the
        title check inside the first `search_limit` hits.

        Raises (does NOT swallow): yt-dlp infrastructure failures
        (DownloadError, network errors, `ImportError` if yt-dlp is
        gone). Per the `AudioSource` Protocol, callers distinguish
        'no coverage' (None) from 'infrastructure broken' (exception)
        so the orchestrator can write a retryable error row rather
        than caching a transient failure as permanent no_coverage.
        """
        ticker_upper = ticker.upper()
        company = TICKER_QUERIES.get(ticker_upper, ticker_upper)
        query = f"{company} earnings call Q{quarter} {year}"

        self.rate_limiter.wait()
        opts = {
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
            "no_warnings": True,
        }
        # No try/except here: infrastructure failures from yt-dlp
        # propagate to the orchestrator's error-row branch. The
        # entry-processing loop below has its own None-filter so a
        # blocked search result doesn't crash this call.
        with self._factory(opts) as ydl:
            info = ydl.extract_info(
                f"ytsearch{self._search_limit}:{query}", download=False
            )

        entries = (info or {}).get("entries") or []
        seen_in_band = 0
        for entry in entries:
            if entry is None:
                # yt-dlp emits None for results that were discovered
                # but couldn't be flat-extracted (private / blocked
                # / age-gated / region-restricted). Skip silently.
                continue
            duration = entry.get("duration")
            if not isinstance(duration, int | float):
                continue
            if not (self._min_duration_sec <= duration <= self._max_duration_sec):
                continue
            seen_in_band += 1
            title = entry.get("title") or ""
            if not _title_matches_call(
                title,
                company=company,
                ticker=ticker_upper,
                year=year,
                quarter=quarter,
            ):
                _logger.debug(
                    "youtube: in-band but title rejected for %s %sQ%s: %r",
                    ticker_upper,
                    year,
                    quarter,
                    title[:120],
                )
                continue
            url = entry.get("webpage_url") or entry.get("url")
            if isinstance(url, str) and url.startswith("https://"):
                return url
        _logger.info(
            "youtube: no title-and-duration match for %s %sQ%s "
            "(query=%r, in_band=%d, total=%d)",
            ticker_upper,
            year,
            quarter,
            query,
            seen_in_band,
            len(entries),
        )
        return None

    def download(self, audio_url: str) -> bytes:
        """Extract the best audio stream at `audio_url` and return its bytes.

        yt-dlp writes the audio to disk; we point it at a
        TemporaryDirectory, validate the output is audio-shaped
        (size floor + magic-byte sniff), then read it back as bytes.
        The temp dir is cleaned up on exit.

        Raises RuntimeError if yt-dlp fails, writes nothing, or
        writes output that doesn't look like audio (typically an
        anti-bot HTML page or JSON challenge). Per the
        `AudioSource` Protocol, these are infrastructure failures —
        the orchestrator routes them to a retryable error row.
        """
        self.rate_limiter.wait()
        with tempfile.TemporaryDirectory(prefix="yt-") as tmpdir:
            outtmpl = str(Path(tmpdir) / "audio.%(ext)s")
            opts = {
                "quiet": True,
                "no_warnings": True,
                # `bestaudio` selects YouTube's highest-quality
                # audio-only stream. We deliberately do NOT fall
                # back to `best` (the full A/V container) — that
                # would silently 4-5x storage and download cost,
                # write a misleading `.bin` filename downstream,
                # and mask format-availability regressions. If no
                # audio-only stream is available, yt-dlp raises
                # and the orchestrator records an error row.
                "format": "bestaudio",
                "outtmpl": outtmpl,
                "noplaylist": True,
                "skip_download": False,
                # Throttle yt-dlp's INTERNAL request volume. Our
                # TokenBucket only sees one wait() per download
                # call, but yt-dlp internally fetches the manifest
                # plus N segments — without these, that's 20-100
                # unthrottled requests to YouTube CDNs per call.
                "sleep_interval": _YTDLP_SLEEP_INTERVAL,
                "max_sleep_interval": _YTDLP_MAX_SLEEP_INTERVAL,
                "sleep_interval_requests": _YTDLP_SLEEP_INTERVAL,
            }
            try:
                with self._factory(opts) as ydl:
                    ydl.extract_info(audio_url, download=True)
            except RuntimeError:
                # Already a typed error from this source — propagate as-is.
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"yt-dlp failed to fetch audio from {audio_url}: {exc}"
                ) from exc

            # Pick the matching audio file by outtmpl pattern (NOT
            # just "largest in tmpdir"): yt-dlp emits exactly one
            # file matching `audio.*` for our config; any other
            # artifact (e.g. .info.json if a future opt enables it)
            # is rejected.
            audio_files = sorted(Path(tmpdir).glob("audio.*"))
            if not audio_files:
                raise RuntimeError(
                    f"yt-dlp completed without writing output for {audio_url}"
                )
            if len(audio_files) > 1:
                # Pick the largest among matching — usually only one,
                # but be defensive about future post-processor changes.
                audio_file = max(audio_files, key=lambda p: p.stat().st_size)
            else:
                audio_file = audio_files[0]

            audio = audio_file.read_bytes()
            if len(audio) < _MIN_AUDIO_BYTES:
                raise RuntimeError(
                    f"yt-dlp output for {audio_url} is suspiciously small "
                    f"({len(audio)} bytes < {_MIN_AUDIO_BYTES} floor); likely an "
                    "error page or partial fetch."
                )
            if not _looks_like_audio(audio[:16]):
                raise RuntimeError(
                    f"yt-dlp output for {audio_url} does not look like audio "
                    f"(first bytes: {audio[:16]!r}); likely an HTML challenge or "
                    "JSON error response."
                )
            return audio


__all__ = [
    "SOURCE_NAME",
    "TICKER_QUERIES",
    "YouTubeSource",
    "YoutubeDLFactory",
]
