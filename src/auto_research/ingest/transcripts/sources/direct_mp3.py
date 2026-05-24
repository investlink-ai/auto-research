"""`direct_mp3` source: companies that post the earnings-call audio
as a plain MP3/M4A link on a per-quarter URL.

The simplest source class. Used as the bootstrap implementation of
the `AudioSource` Protocol — heavier sources (YouTube via yt-dlp,
future platforms) follow the same shape with more discovery /
download machinery inside `find_audio_url` and `download`.

Coverage in the universe is uncertain. A static IR-probe found a few
candidates among smaller industrial caps, but Cloudflare blocked the
most-promising ones; the coverage-survey worker populates the
registry with whatever tickers actually use this pattern. Until then
this source is the canonical reference implementation rather than a
coverage workhorse.
"""

from __future__ import annotations

from typing import Final

import httpx

from auto_research.ingest import _http
from auto_research.ingest.rate_limit import TokenBucket
from auto_research.ingest.transcripts._base import TranscriptConfigError
from auto_research.ingest.transcripts._config import load_sources_config

SOURCE_NAME: Final = "direct_mp3"
SOURCE_LABEL: Final = "direct-MP3"

# Per-ticker URL templates. Loaded from `config/transcripts/sources.
# toml` (the `[tickers]` table's `url` field on rows whose source is
# `direct_mp3`). Each value is a Python format string with named
# placeholders `{year}` and `{quarter}` — the source substitutes
# them at call time. URLs themselves are checked-in data (not
# scraped discovery), kept in config because each ticker's IR
# layout is bespoke: `https://investor.acme.com/audio/2024Q1-
# earnings.mp3` etc.
TICKER_URL_TEMPLATES: dict[str, str] = {
    ticker: cfg.url
    for ticker, cfg in load_sources_config().tickers.items()
    if cfg.source == SOURCE_NAME and cfg.url is not None
}


class DirectMp3Source:
    """Fetches a static MP3/M4A URL per ticker x (year, quarter).

    Reuses the EDGAR rate-limit + retry + classify discipline from
    `auto_research.ingest._http`. The default `TokenBucket` is 8 r/s
    (same as SEC) because each IR host is rate-sensitive in its own
    way; tighten per-host later if any one ticker gets aggressive.
    """

    name = SOURCE_NAME

    def __init__(
        self,
        *,
        rate_limiter: TokenBucket | None = None,
        timeout: float = 60.0,
        max_attempts: int = _http.DEFAULT_MAX_ATTEMPTS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # Default rate limiter is intentionally NOT the SEC one — this
        # source talks to issuer IR hosts, not SEC. 8 r/s is just our
        # generic "be polite" baseline.
        self.rate_limiter = rate_limiter or TokenBucket(rate=8.0)
        self._max_attempts = max_attempts
        self._client = httpx.Client(
            headers=_http.default_headers(),  # no UA needed; IR pages don't gate on it
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DirectMp3Source:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def find_audio_url(self, ticker: str, year: int, quarter: int) -> str | None:
        """Resolve the static URL template for this ticker x quarter.

        Returns None for tickers not in `TICKER_URL_TEMPLATES`. The
        orchestrator treats None as 'no coverage' and writes the
        corresponding manifest row.
        """
        template = TICKER_URL_TEMPLATES.get(ticker.upper())
        if template is None:
            return None
        try:
            return template.format(year=year, quarter=quarter)
        except KeyError as exc:
            raise TranscriptConfigError(
                f"TICKER_URL_TEMPLATES['{ticker.upper()}'] references "
                f"unknown placeholder {exc.args[0]!r}; expected only "
                f"{{year}} and {{quarter}}."
            ) from exc

    def download(self, audio_url: str) -> bytes:
        """Rate-limited GET of `audio_url`, returning raw bytes.

        Retries on the standard transient set (429, 5xx, empty-200,
        network errors) via tenacity + `_http.make_retry_wait`.
        """
        from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

        retryable = _http.retryable_exceptions(
            rate_limited=_http.RateLimited,
            server_error=_http.ServerError,
            empty_response=_http.EmptyResponseError,
        )
        for attempt in Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=_http.make_retry_wait(),
            retry=retry_if_exception_type(retryable),
            reraise=True,
        ):
            with attempt:
                self.rate_limiter.wait()
                resp = self._client.get(audio_url)
                _http.classify_response(
                    resp,
                    rate_limited=_http.RateLimited,
                    server_error=_http.ServerError,
                    empty_response=_http.EmptyResponseError,
                    source_label=SOURCE_LABEL,
                )
                resp.raise_for_status()
                return resp.content
        raise RuntimeError("unreachable: tenacity always returns or raises")


__all__ = ["SOURCE_NAME", "TICKER_URL_TEMPLATES", "DirectMp3Source"]
