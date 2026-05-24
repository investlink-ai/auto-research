"""Unit tests for the `youtube` source.

The real `yt_dlp.YoutubeDL` never runs — `YouTubeSource` accepts a
`factory` kwarg whose default produces a real client. Tests pass a
fake factory whose `extract_info` returns canned search results
(for `find_audio_url`) or simulates a download by writing bytes to
the `outtmpl` directory (for `download`).

A live-smoke test that exercises real yt-dlp against the real
YouTube backend lives in `tests/live/`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from auto_research.ingest.rate_limit import TokenBucket
from auto_research.ingest.transcripts.sources import youtube
from auto_research.ingest.transcripts.sources.youtube import YouTubeSource


def _fast_limiter() -> TokenBucket:
    return TokenBucket(rate=1000.0, capacity=1000.0)


class _FakeYDL:
    """Captures opts + serves canned extract_info responses.

    For SEARCH calls (`url` starts with `ytsearch`): returns the
    `search_response` dict (typically `{"entries": [...]}`).
    For DOWNLOAD calls (any other URL): writes `audio_bytes` to the
    output template path then returns a minimal info dict.
    """

    def __init__(
        self,
        opts: dict[str, Any],
        *,
        search_response: dict[str, Any] | None = None,
        audio_bytes: bytes | None = None,
        download_raises: Exception | None = None,
    ) -> None:
        self.opts = opts
        self._search_response = search_response or {"entries": []}
        self._audio_bytes = audio_bytes or b""
        self._download_raises = download_raises
        self.calls: list[tuple[str, bool]] = []

    def __enter__(self) -> _FakeYDL:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def extract_info(
        self, url: str, *, download: bool = False
    ) -> dict[str, Any] | None:
        self.calls.append((url, download))
        if url.startswith("ytsearch"):
            return self._search_response
        if self._download_raises:
            raise self._download_raises
        # Simulate the "download" step by writing bytes to the
        # outtmpl path (yt-dlp's real behavior).
        outtmpl = self.opts.get("outtmpl", "")
        if isinstance(outtmpl, str) and outtmpl:
            # outtmpl looks like ".../audio.%(ext)s"; substitute ext=m4a.
            path = Path(outtmpl.replace("%(ext)s", "m4a"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._audio_bytes)
        return {"id": "fake", "ext": "m4a"}


def _factory(**kwargs: Any) -> Callable[[dict[str, Any]], _FakeYDL]:
    """Build a factory that yields a fresh _FakeYDL with shared canned state."""

    def make(opts: dict[str, Any]) -> _FakeYDL:
        return _FakeYDL(opts, **kwargs)

    return make


# ---------- find_audio_url ----------


def test_find_audio_url_returns_first_in_band_match() -> None:
    """Out of N search results, return the first whose duration lies
    in the configured earnings-call band."""
    response = {
        "entries": [
            {"duration": 200, "url": "https://youtube.com/watch?v=short"},
            {"duration": 3997, "url": "https://youtube.com/watch?v=full1"},
            {"duration": 4108, "url": "https://youtube.com/watch?v=full2"},
        ]
    }
    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=_factory(search_response=response),
    )
    assert src.find_audio_url("NVDA", 2024, 2) == "https://youtube.com/watch?v=full1"


def test_find_audio_url_falls_back_to_webpage_url() -> None:
    """If `url` is missing (some yt-dlp versions/options return only
    `webpage_url`), use that instead."""
    response = {
        "entries": [
            {"duration": 3500, "webpage_url": "https://youtube.com/watch?v=fallback"},
        ]
    }
    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=_factory(search_response=response),
    )
    assert (
        src.find_audio_url("NVDA", 2024, 2) == "https://youtube.com/watch?v=fallback"
    )


def test_find_audio_url_returns_none_when_no_in_band_match() -> None:
    """All results too short / too long → coverage gap, not crash."""
    response = {
        "entries": [
            {"duration": 100, "url": "https://youtube.com/watch?v=clip"},
            {"duration": 50000, "url": "https://youtube.com/watch?v=longstream"},
        ]
    }
    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=_factory(search_response=response),
    )
    assert src.find_audio_url("NVDA", 2024, 2) is None


def test_find_audio_url_returns_none_on_empty_results() -> None:
    """No search hits → None."""
    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=_factory(search_response={"entries": []}),
    )
    assert src.find_audio_url("NEVER_TRADED", 2024, 2) is None


def test_find_audio_url_returns_none_on_factory_exception() -> None:
    """A yt-dlp failure (network, parse error, etc.) surfaces as None,
    not a raised exception that would break a batch worker loop."""

    def boom(opts: dict[str, Any]) -> _FakeYDL:
        raise RuntimeError("network down")

    src = YouTubeSource(rate_limiter=_fast_limiter(), factory=boom)
    assert src.find_audio_url("NVDA", 2024, 2) is None


def test_find_audio_url_uses_company_override_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`TICKER_QUERIES` override replaces the ticker in the query so
    aggregator titles like 'NVIDIA Q2 ...' match instead of clashing
    with crypto / unrelated content for short symbols."""
    monkeypatch.setitem(youtube.TICKER_QUERIES, "ACME", "Acme Corporation")
    seen_queries: list[str] = []

    def capturing_factory(opts: dict[str, Any]) -> _FakeYDL:
        class _Capturing(_FakeYDL):
            def extract_info(self, url: str, *, download: bool = False):
                seen_queries.append(url)
                return {"entries": [{"duration": 3600, "url": "https://x"}]}

        return _Capturing(opts)

    src = YouTubeSource(rate_limiter=_fast_limiter(), factory=capturing_factory)
    src.find_audio_url("ACME", 2024, 3)
    assert any("Acme Corporation" in q for q in seen_queries)
    # And the ticker symbol must NOT leak when overridden.
    assert not any("ACME" in q and "Acme" not in q for q in seen_queries)


def test_find_audio_url_falls_back_to_ticker_when_no_override() -> None:
    """No override → query uses the ticker symbol verbatim."""
    seen_queries: list[str] = []

    def capturing_factory(opts: dict[str, Any]) -> _FakeYDL:
        class _Capturing(_FakeYDL):
            def extract_info(self, url: str, *, download: bool = False):
                seen_queries.append(url)
                return {"entries": []}

        return _Capturing(opts)

    src = YouTubeSource(rate_limiter=_fast_limiter(), factory=capturing_factory)
    src.find_audio_url("ZZTOP", 2024, 2)
    assert any("ZZTOP" in q for q in seen_queries)


def test_find_audio_url_is_case_insensitive() -> None:
    response = {
        "entries": [{"duration": 3500, "url": "https://youtube.com/watch?v=hit"}]
    }
    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=_factory(search_response=response),
    )
    assert src.find_audio_url("nvda", 2024, 2) == "https://youtube.com/watch?v=hit"


# ---------- download ----------


def test_download_returns_audio_bytes() -> None:
    """`download` invokes yt-dlp's extractor and returns the audio
    file's bytes."""
    expected = b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 1000  # m4a-ish magic
    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=_factory(audio_bytes=expected),
    )
    audio = src.download("https://youtube.com/watch?v=fake")
    assert audio == expected


def test_download_raises_runtime_error_when_no_file_written() -> None:
    """yt-dlp returned without writing to the outtmpl path → error
    (rather than returning b'' which would look like a no_coverage)."""

    def no_op_factory(opts: dict[str, Any]) -> _FakeYDL:
        class _NoOp(_FakeYDL):
            def extract_info(self, url: str, *, download: bool = False):
                # No bytes written to outtmpl.
                return {"id": "fake"}

        return _NoOp(opts)

    src = YouTubeSource(rate_limiter=_fast_limiter(), factory=no_op_factory)
    with pytest.raises(RuntimeError, match="without writing output"):
        src.download("https://youtube.com/watch?v=fake")


def test_download_wraps_factory_exception() -> None:
    """A yt-dlp DownloadError or similar surfaces as RuntimeError with
    operator-actionable context (the URL it was trying to fetch)."""
    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=_factory(download_raises=ValueError("HTTP 403")),
    )
    with pytest.raises(RuntimeError, match="403"):
        src.download("https://youtube.com/watch?v=blocked")


def test_download_returns_largest_file_when_multiple_emitted(
    tmp_path: Path,
) -> None:
    """yt-dlp occasionally writes a .info.json alongside the audio.
    Return the largest file (the actual audio), not the metadata."""

    audio = b"\x00\xff" * 5000  # 10KB of bytes

    def make(opts: dict[str, Any]) -> _FakeYDL:
        outtmpl = opts.get("outtmpl", "")

        class _Multi(_FakeYDL):
            def extract_info(self, url: str, *, download: bool = False):
                # Write a tiny info file + the real audio.
                base = Path(outtmpl.replace("%(ext)s", "m4a"))
                base.parent.mkdir(parents=True, exist_ok=True)
                base.write_bytes(audio)
                (base.parent / "audio.info.json").write_bytes(b'{"id":"x"}')
                return {"id": "x"}

        return _Multi(opts)

    src = YouTubeSource(rate_limiter=_fast_limiter(), factory=make)
    assert src.download("https://y/watch") == audio


# ---------- construction ----------


def test_constructor_rejects_zero_search_limit() -> None:
    with pytest.raises(ValueError, match="search_limit"):
        YouTubeSource(search_limit=0)


def test_constructor_rejects_inverted_duration_band() -> None:
    with pytest.raises(ValueError, match="duration band"):
        YouTubeSource(min_duration_sec=6000, max_duration_sec=3000)


# ---------- source identity / registry shape ----------


def test_source_name_is_youtube() -> None:
    assert YouTubeSource.name == youtube.SOURCE_NAME == "youtube"


def test_nvda_registered_to_youtube() -> None:
    """NVDA is the canary; its registry entry must point to `youtube`."""
    from auto_research.ingest.transcripts import registry

    assert registry.REGISTRY["NVDA"] == "youtube"


def test_nvda_has_query_override() -> None:
    """The canary's query override must be configured; otherwise the
    ticker symbol clashes with unrelated content on YouTube."""
    assert "NVDA" in youtube.TICKER_QUERIES
    assert youtube.TICKER_QUERIES["NVDA"].strip()
