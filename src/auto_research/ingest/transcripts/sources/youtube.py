"""`youtube` source: earnings-call audio via yt-dlp.

Most large- and mid-cap earnings calls are mirrored on YouTube within
hours by aggregator channels (Benzinga, Castify Earnings Call, EARNMOAR,
Investing 101, Yahoo Finance) plus the occasional first-party upload.
A universe-wide probe of the 81-ticker v1 universe found a full-length
match (40-90 min duration, the consistent earnings-call shape) for
every ticker when queried by company name. yt-dlp fetches the audio
stream directly — no headless browser, no platform registration.

`find_audio_url` runs a `ytsearch{N}:{query}` lookup, filters results
by duration band, and returns the first match's video URL.
`download` invokes yt-dlp again to extract the best audio stream as
bytes (typically m4a/AAC).

Search query: by default `{ticker} earnings call`, but per-ticker
overrides in `TICKER_QUERIES` let us supply the company name when
the ticker symbol clashes with unrelated YouTube content (a real
problem for tickers like MIR / LEU / FORM whose tickers collide with
crypto and other topics — see the coverage survey notes).

Test seam: `YoutubeDLFactory` returns a `yt_dlp.YoutubeDL`-like object
with `extract_info`. Production uses real yt-dlp; tests inject a fake
that yields canned search results + download bytes.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol

from auto_research.ingest.rate_limit import TokenBucket

SOURCE_NAME: Final = "youtube"

# Per-ticker search query override. The default query is
# `{ticker} earnings call`, which works for most tickers but
# false-positives on short tickers whose symbols collide with crypto
# tokens, file extensions, etc. Override here when the ticker's
# symbol is ambiguous; the value is appended with ` earnings call`.
#
# Adding a ticker = adding a row here. The Playwright re-survey in the
# next batch worker iteration is the canonical place to populate this
# en masse; one entry is seeded as a canary for the live path.
TICKER_QUERIES: dict[str, str] = {
    "NVDA": "NVIDIA",
}

# Earnings calls are typically 40-90 minutes (2400-5400 sec). We use a
# slightly wider band (30-100 min) to absorb shorter calls (small caps,
# pre-recorded statements) and longer ones (with extended Q&A).
_MIN_DURATION_SEC: Final = 1800
_MAX_DURATION_SEC: Final = 6000

# Default number of search results to retrieve per query. The
# duration filter then picks the first qualifying result. 5 is
# enough for the universe: aggregators usually appear in the top 3.
_DEFAULT_SEARCH_LIMIT: Final = 5

# Rate budget for YouTube requests. yt-dlp aggregates many fetches per
# call (search + format negotiation + chunked download). 2 r/s is
# conservative — YouTube hasn't published an official rate ceiling
# for the unauthenticated paths yt-dlp uses, but anti-automation
# pressure grows linearly with volume.
_DEFAULT_RATE_PER_SEC: Final = 2.0

_logger = logging.getLogger(__name__)


class _YoutubeDLLike(Protocol):
    """The narrow yt-dlp surface this source uses.

    Real `yt_dlp.YoutubeDL` satisfies this Protocol structurally; tests
    can inject any class with `extract_info()` returning a dict.
    """

    def extract_info(
        self, url: str, *, download: bool = False
    ) -> dict[str, Any] | None: ...


# yt-dlp opens in a context-manager idiom (`with YoutubeDL(opts) as ydl:`),
# but `YoutubeDL(opts)` is also valid as a direct constructor. The factory
# returns a fresh client each call; the source closes it via `__exit__`.
YoutubeDLFactory = Callable[[dict[str, Any]], _YoutubeDLLike]


def _default_factory(opts: dict[str, Any]) -> _YoutubeDLLike:
    """Production yt-dlp factory. Lazy import — keeps `yt_dlp` out of
    the import graph for unit tests that inject a fake."""
    from yt_dlp import YoutubeDL  # type: ignore[import-untyped]

    client: _YoutubeDLLike = YoutubeDL(opts)
    return client


class YouTubeSource:
    """Earnings audio for any ticker whose call gets mirrored on YouTube.

    `find_audio_url` runs `ytsearch{N}:{query}` and returns the first
    result whose duration falls in the earnings-call band (30-100 min).
    `download` invokes yt-dlp's audio extractor and returns the file
    bytes for the orchestrator to persist.
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
    ) -> None:
        if search_limit < 1:
            raise ValueError(f"search_limit must be >= 1, got {search_limit}")
        if min_duration_sec < 0 or max_duration_sec <= min_duration_sec:
            raise ValueError(
                f"invalid duration band: ({min_duration_sec}, {max_duration_sec})"
            )
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
        """Return a YouTube URL whose duration matches a full earnings call.

        The query is `{TICKER_QUERIES.get(ticker, ticker)} earnings call
        Q{quarter} {year}`. We add the quarter/year to bias the search
        toward the right call; YouTube's relevance ranking does the rest.
        Returns None if no result in the duration band shows up in the
        first `search_limit` hits.
        """
        ticker_upper = ticker.upper()
        company = TICKER_QUERIES.get(ticker_upper, ticker_upper)
        query = f"{company} earnings call Q{quarter} {year}"

        self.rate_limiter.wait()
        opts = {"quiet": True, "extract_flat": True, "skip_download": True}
        try:
            with self._factory(opts) as ydl:  # type: ignore[attr-defined]
                info = ydl.extract_info(
                    f"ytsearch{self._search_limit}:{query}", download=False
                )
        except Exception as exc:
            _logger.warning(
                "youtube search failed for %s %sQ%s: %s", ticker_upper, year, quarter, exc
            )
            return None

        entries = (info or {}).get("entries") or []
        for entry in entries:
            duration = entry.get("duration")
            if not isinstance(duration, int | float):
                continue
            if self._min_duration_sec <= duration <= self._max_duration_sec:
                url = entry.get("url") or entry.get("webpage_url")
                if isinstance(url, str) and url:
                    return url
        _logger.info(
            "youtube: no in-band match for %s %sQ%s (query=%r, results=%d)",
            ticker_upper,
            year,
            quarter,
            query,
            len(entries),
        )
        return None

    def download(self, audio_url: str) -> bytes:
        """Extract the best audio stream at `audio_url` and return its bytes.

        yt-dlp writes the audio to disk by default; we point it at a
        TemporaryDirectory, read the result back as bytes, and let the
        context manager clean up the tmp file.
        """
        self.rate_limiter.wait()
        with tempfile.TemporaryDirectory(prefix="yt-") as tmpdir:
            outtmpl = str(Path(tmpdir) / "audio.%(ext)s")
            opts = {
                "quiet": True,
                "no_warnings": True,
                # `bestaudio` selects the highest-quality audio-only
                # stream YouTube offers. Falling back to `best` picks
                # an A/V format whose audio track is then extracted.
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "noplaylist": True,
                "skip_download": False,
                "extract_audio": False,
            }
            try:
                with self._factory(opts) as ydl:  # type: ignore[attr-defined]
                    ydl.extract_info(audio_url, download=True)
            except Exception as exc:
                raise RuntimeError(
                    f"yt-dlp failed to fetch audio from {audio_url}: {exc}"
                ) from exc

            files = list(Path(tmpdir).iterdir())
            if not files:
                raise RuntimeError(
                    f"yt-dlp completed without writing output for {audio_url}"
                )
            # `bestaudio` typically emits one file. If multiple appear
            # (rare — happens when yt-dlp post-processes), pick the
            # largest, which is the actual audio (versus a tiny .info.json).
            audio_file = max(files, key=lambda p: p.stat().st_size)
            return audio_file.read_bytes()


__all__ = [
    "SOURCE_NAME",
    "TICKER_QUERIES",
    "YouTubeSource",
    "YoutubeDLFactory",
]
