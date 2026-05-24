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
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest

from auto_research.ingest.rate_limit import TokenBucket
from auto_research.ingest.transcripts._base import TranscriptConfigError
from auto_research.ingest.transcripts.sources import youtube
from auto_research.ingest.transcripts.sources.youtube import (
    YouTubeSource,
    _looks_like_audio,
    _title_matches_call,
    _year_tokens,
)


def _fast_limiter() -> TokenBucket:
    return TokenBucket(rate=1000.0, capacity=1000.0)


# Magic-byte-valid audio payload for download tests. Real m4a starts
# with a 4-byte size + `ftyp` at offset 4; we follow that shape and
# pad to clear the 100KB size floor.
_VALID_M4A_HEADER = b"\x00\x00\x00\x18ftypM4A "
_VALID_AUDIO_BYTES = _VALID_M4A_HEADER + b"\x00" * 200_000  # ~200KB


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
        extract_raises: Exception | None = None,
    ) -> None:
        self.opts = opts
        self._search_response = search_response or {"entries": []}
        self._audio_bytes = audio_bytes
        self._download_raises = download_raises
        self._extract_raises = extract_raises
        self.calls: list[tuple[str, bool]] = []

    def __enter__(self) -> _FakeYDL:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def extract_info(
        self, url: str, *, download: bool = False
    ) -> dict[str, Any] | None:
        self.calls.append((url, download))
        if self._extract_raises:
            raise self._extract_raises
        if url.startswith("ytsearch"):
            return self._search_response
        if self._download_raises:
            raise self._download_raises
        if self._audio_bytes is None:
            # Simulate yt-dlp's real failure shape: no file written.
            return {"id": "fake", "ext": "m4a"}
        outtmpl = self.opts.get("outtmpl", "")
        if isinstance(outtmpl, str) and outtmpl:
            path = Path(outtmpl.replace("%(ext)s", "m4a"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._audio_bytes)
        return {"id": "fake", "ext": "m4a"}


def _factory(
    **kwargs: Any,
) -> Callable[[dict[str, Any]], AbstractContextManager[_FakeYDL]]:
    """Build a factory that yields a fresh _FakeYDL with shared canned state."""

    def make(opts: dict[str, Any]) -> _FakeYDL:
        return _FakeYDL(opts, **kwargs)

    return make


def _make_source(**factory_kwargs: Any) -> YouTubeSource:
    """Construct a YouTubeSource with verify_yt_dlp=False so unit
    tests don't depend on yt-dlp actually being importable."""
    return YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=_factory(**factory_kwargs),
        verify_yt_dlp=False,
    )


def _entry(
    *,
    title: str,
    duration: int | float = 3600,
    url: str | None = None,
    webpage_url: str | None = None,
) -> dict[str, Any]:
    """Build a yt-dlp-shaped search entry."""
    out: dict[str, Any] = {"title": title, "duration": duration}
    if url is not None:
        out["url"] = url
    if webpage_url is not None:
        out["webpage_url"] = webpage_url
    return out


# ---------- _looks_like_audio ----------


@pytest.mark.parametrize(
    "head",
    [
        b"ID3" + b"\x00" * 20,
        b"OggS" + b"\x00" * 20,
        b"RIFF" + b"\x00" * 20,
        b"\x1a\x45\xdf\xa3" + b"\x00" * 20,
        b"fLaC" + b"\x00" * 20,
        b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 8,  # MP4/M4A ftyp at offset 4
        b"\xff\xfb" + b"\x00" * 20,  # MP3 ADTS sync 0xFFFB
        b"\xff\xf1" + b"\x00" * 20,  # AAC ADTS sync 0xFFF1
    ],
)
def test_looks_like_audio_accepts_known_containers(head: bytes) -> None:
    assert _looks_like_audio(head) is True


@pytest.mark.parametrize(
    "head",
    [
        b"<!DOCTYPE html><html>",  # HTML challenge page
        b'{"error":"forbidden"}',  # JSON error
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,  # PNG (thumbnail)
        b"GIF89a" + b"\x00" * 8,
        b"",
        b"\x00",
    ],
)
def test_looks_like_audio_rejects_non_audio(head: bytes) -> None:
    assert _looks_like_audio(head) is False


# ---------- _year_tokens ----------


def test_year_tokens_includes_calendar_and_fiscal_framings() -> None:
    toks = _year_tokens(2026)
    assert "2026" in toks
    assert "2025" in toks  # prior year (for fiscal-Q calls ending earlier)
    assert "fy26" in toks
    assert "fy25" in toks
    # Long-form FY also accepted (some aggregators write 'FY2026').
    assert "fy2026" in toks


# ---------- _title_matches_call ----------


def test_title_matches_canonical_aggregator_format() -> None:
    """Benzinga / Castify titles use 'COMPANY Q{N} FY{NN} Earnings Call'."""
    assert _title_matches_call(
        "NVIDIA Q4 FY26 Earnings Call",
        company="NVIDIA",
        ticker="NVDA",
        year=2026,
        quarter=4,
    )


def test_title_matches_calendar_year_format() -> None:
    """Some aggregators use 'Q4 2026' for the same call."""
    assert _title_matches_call(
        "NVIDIA Q4 2026 Earnings Conference Call",
        company="NVIDIA",
        ticker="NVDA",
        year=2026,
        quarter=4,
    )


def test_title_matches_prior_year_for_fiscal_q4() -> None:
    """A fiscal Q4 ending in early 2026 can be labeled by the prior
    calendar year — 'Q4 2025' should still match if the caller is
    asking for the FY26 framing."""
    assert _title_matches_call(
        "NVIDIA Q4 2025 Earnings Call",
        company="NVIDIA",
        ticker="NVDA",
        year=2026,
        quarter=4,
    )


def test_title_rejects_wrong_quarter() -> None:
    """A Q3 video must NOT match a Q4 request even with matching
    company + year tokens (this is the core silent-correctness gate)."""
    assert not _title_matches_call(
        "NVIDIA Q3 FY26 Earnings Call",
        company="NVIDIA",
        ticker="NVDA",
        year=2026,
        quarter=4,
    )


def test_title_rejects_wrong_company() -> None:
    assert not _title_matches_call(
        "Apple Q4 FY26 Earnings Call",
        company="NVIDIA",
        ticker="NVDA",
        year=2026,
        quarter=4,
    )


def test_title_rejects_wrong_year_framing() -> None:
    """A 2024 call must not match a 2026 request even with same Q."""
    assert not _title_matches_call(
        "NVIDIA Q4 FY24 Earnings Call",
        company="NVIDIA",
        ticker="NVDA",
        year=2026,
        quarter=4,
    )


def test_title_accepts_ticker_dollar_prefix() -> None:
    """'$NVDA Q4 2026 Earnings Call' (no company name) still passes."""
    assert _title_matches_call(
        "$NVDA Q4 2026 Earnings Call",
        company="NVIDIA",
        ticker="NVDA",
        year=2026,
        quarter=4,
    )


# ---------- find_audio_url ----------


def test_find_audio_url_returns_first_title_and_duration_match() -> None:
    """Pick the first entry that passes BOTH the duration band AND
    the title check (company+quarter+year)."""
    response = {
        "entries": [
            _entry(
                title="Random clip about NVIDIA stock",
                duration=200,
                webpage_url="https://youtube.com/watch?v=short",
            ),
            _entry(
                title="NVIDIA Q4 FY26 Earnings Conference Call",
                duration=3997,
                webpage_url="https://youtube.com/watch?v=correct",
            ),
            _entry(
                title="NVIDIA Q4 FY26 Earnings Recap",
                duration=4108,
                webpage_url="https://youtube.com/watch?v=second",
            ),
        ]
    }
    src = _make_source(search_response=response)
    assert (
        src.find_audio_url("NVDA", 2026, 4)
        == "https://youtube.com/watch?v=correct"
    )


def test_find_audio_url_skips_in_band_when_title_doesnt_match() -> None:
    """An in-band video whose title is for the WRONG ticker must be
    rejected — this is the core wrong-ticker-match defense."""
    response = {
        "entries": [
            _entry(
                title="Microsoft Q4 FY26 Earnings Call",
                duration=3997,
                webpage_url="https://youtube.com/watch?v=wrongticker",
            ),
            _entry(
                title="NVIDIA Q4 FY26 Earnings Call",
                duration=3500,
                webpage_url="https://youtube.com/watch?v=correct",
            ),
        ]
    }
    src = _make_source(search_response=response)
    assert (
        src.find_audio_url("NVDA", 2026, 4)
        == "https://youtube.com/watch?v=correct"
    )


def test_find_audio_url_skips_in_band_when_quarter_wrong() -> None:
    """An in-band video whose title is for the WRONG quarter must be
    rejected — the wrong-quarter-match defense."""
    response = {
        "entries": [
            _entry(
                title="NVIDIA Q3 FY26 Earnings Call",
                duration=3997,
                webpage_url="https://youtube.com/watch?v=q3",
            ),
        ]
    }
    src = _make_source(search_response=response)
    assert src.find_audio_url("NVDA", 2026, 4) is None


def test_find_audio_url_skips_none_entries() -> None:
    """yt-dlp emits None for private/blocked/age-gated results. The
    loop must skip them rather than crashing with AttributeError."""
    response = {
        "entries": [
            None,  # blocked
            None,  # private
            _entry(
                title="NVIDIA Q4 FY26 Earnings Call",
                duration=3500,
                webpage_url="https://youtube.com/watch?v=ok",
            ),
        ]
    }
    src = _make_source(search_response=response)
    assert src.find_audio_url("NVDA", 2026, 4) == "https://youtube.com/watch?v=ok"


def test_find_audio_url_returns_none_when_no_in_band_match() -> None:
    response = {
        "entries": [
            _entry(
                title="NVIDIA Q4 FY26 Earnings Call",
                duration=100,
                webpage_url="https://youtube.com/watch?v=clip",
            ),
            _entry(
                title="NVIDIA Q4 FY26 Earnings Call",
                duration=50000,
                webpage_url="https://youtube.com/watch?v=longstream",
            ),
        ]
    }
    src = _make_source(search_response=response)
    assert src.find_audio_url("NVDA", 2026, 4) is None


def test_find_audio_url_returns_none_on_empty_results() -> None:
    src = _make_source(search_response={"entries": []})
    assert src.find_audio_url("NEVER_TRADED", 2024, 2) is None


def test_find_audio_url_propagates_factory_exception() -> None:
    """Infrastructure failure (network, yt-dlp crash) must propagate —
    NOT be swallowed and converted to None (which would land in a
    permanent no_coverage cache row). The orchestrator converts
    these into retryable error rows."""

    def boom(opts: dict[str, Any]) -> _FakeYDL:
        raise RuntimeError("network down")

    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=boom,
        verify_yt_dlp=False,
    )
    with pytest.raises(RuntimeError, match="network down"):
        src.find_audio_url("NVDA", 2026, 4)


def test_find_audio_url_propagates_extract_info_exception() -> None:
    """Same: yt-dlp raising from inside extract_info propagates."""
    src = _make_source(extract_raises=RuntimeError("YouTube returned 403"))
    with pytest.raises(RuntimeError, match="403"):
        src.find_audio_url("NVDA", 2026, 4)


def test_find_audio_url_uses_company_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TICKER_QUERIES override replaces the ticker in the query so
    aggregator titles like 'Acme Corp Q3 2024' match."""
    monkeypatch.setitem(youtube.TICKER_QUERIES, "ACME", "Acme Corporation")
    seen_queries: list[str] = []

    def capturing_factory(
        opts: dict[str, Any],
    ) -> _FakeYDL:
        class _Capturing(_FakeYDL):
            def extract_info(
                self, url: str, *, download: bool = False
            ) -> dict[str, Any] | None:
                seen_queries.append(url)
                return {
                    "entries": [
                        _entry(
                            title="Acme Corporation Q3 FY24 Earnings Call",
                            duration=3600,
                            webpage_url="https://youtube.com/watch?v=acme",
                        )
                    ]
                }

        return _Capturing(opts)

    src = YouTubeSource(
        rate_limiter=_fast_limiter(),
        factory=capturing_factory,
        verify_yt_dlp=False,
    )
    src.find_audio_url("ACME", 2024, 3)
    assert any("Acme Corporation" in q for q in seen_queries)


def test_find_audio_url_is_case_insensitive() -> None:
    response = {
        "entries": [
            _entry(
                title="NVIDIA Q4 FY26 Earnings Call",
                duration=3500,
                webpage_url="https://youtube.com/watch?v=hit",
            )
        ]
    }
    src = _make_source(search_response=response)
    assert src.find_audio_url("nvda", 2026, 4) == "https://youtube.com/watch?v=hit"


def test_find_audio_url_requires_https_url() -> None:
    """Entries whose `webpage_url` is something other than https are
    rejected — defends against future yt-dlp behavior changes that
    might return bare video IDs."""
    response = {
        "entries": [
            _entry(
                title="NVIDIA Q4 FY26 Earnings Call",
                duration=3500,
                url="abc123def",  # bare video ID, no scheme
            )
        ]
    }
    src = _make_source(search_response=response)
    assert src.find_audio_url("NVDA", 2026, 4) is None


# ---------- download ----------


def test_download_returns_audio_bytes() -> None:
    src = _make_source(audio_bytes=_VALID_AUDIO_BYTES)
    audio = src.download("https://youtube.com/watch?v=fake")
    assert audio == _VALID_AUDIO_BYTES


def test_download_raises_when_no_file_written() -> None:
    """yt-dlp returned without writing the outtmpl path → loud error.
    Real yt-dlp would never silently emit nothing on success."""
    src = _make_source(audio_bytes=None)
    with pytest.raises(RuntimeError, match="without writing output"):
        src.download("https://youtube.com/watch?v=fake")


def test_download_raises_below_size_floor() -> None:
    """A few-KB payload is almost certainly an HTML error page or
    partial fetch, not real earnings audio."""
    tiny = _VALID_M4A_HEADER + b"\x00" * 100  # passes magic check, fails size
    src = _make_source(audio_bytes=tiny)
    with pytest.raises(RuntimeError, match="suspiciously small"):
        src.download("https://youtube.com/watch?v=fake")


def test_download_raises_when_output_not_audio() -> None:
    """A large file that doesn't pass the magic-byte sniff (HTML
    error page, JSON challenge) must be rejected — feeding it to
    Whisper would burn API budget on garbage."""
    fake_html = b"<!DOCTYPE html><html><body>blocked</body>" + b"\x00" * 200_000
    src = _make_source(audio_bytes=fake_html)
    with pytest.raises(RuntimeError, match="does not look like audio"):
        src.download("https://youtube.com/watch?v=fake")


def test_download_wraps_factory_exception() -> None:
    src = _make_source(download_raises=ValueError("HTTP 403"))
    with pytest.raises(RuntimeError, match="403"):
        src.download("https://youtube.com/watch?v=blocked")


def test_download_picks_largest_among_audio_glob() -> None:
    """The audio.* glob excludes any non-matching artifact (e.g.
    info.json) yt-dlp might write; among multiple audio.* matches,
    pick the largest."""

    def make(opts: dict[str, Any]) -> _FakeYDL:
        outtmpl = opts.get("outtmpl", "")
        # Drop the audio file plus a phantom non-matching artifact
        # to verify the glob filters it out.
        base = Path(outtmpl.replace("%(ext)s", "m4a"))
        base.parent.mkdir(parents=True, exist_ok=True)

        class _Multi(_FakeYDL):
            def extract_info(
                self, url: str, *, download: bool = False
            ) -> dict[str, Any] | None:
                base.write_bytes(_VALID_AUDIO_BYTES)
                # info.json sibling — NOT matched by audio.* glob.
                (base.parent / "metadata.json").write_bytes(b'{"x":1}')
                return {"id": "x"}

        return _Multi(opts)

    src = YouTubeSource(
        rate_limiter=_fast_limiter(), factory=make, verify_yt_dlp=False
    )
    assert src.download("https://y/watch") == _VALID_AUDIO_BYTES


# ---------- construction ----------


def test_constructor_rejects_zero_search_limit() -> None:
    with pytest.raises(ValueError, match="search_limit"):
        YouTubeSource(search_limit=0, verify_yt_dlp=False)


def test_constructor_rejects_inverted_duration_band() -> None:
    with pytest.raises(ValueError, match="duration band"):
        YouTubeSource(
            min_duration_sec=6000,
            max_duration_sec=3000,
            verify_yt_dlp=False,
        )


def test_constructor_raises_if_yt_dlp_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing yt-dlp install → TranscriptConfigError at construction,
    NOT a silent no_coverage row mid-fetch."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yt_dlp":
            raise ImportError("No module named 'yt_dlp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TranscriptConfigError, match="yt-dlp is not installed"):
        YouTubeSource()


def test_constructor_skips_yt_dlp_check_when_disabled() -> None:
    """Tests can bypass the import probe via verify_yt_dlp=False."""
    src = YouTubeSource(verify_yt_dlp=False)
    assert src.name == "youtube"


# ---------- source identity / registry shape ----------


def test_source_name_is_youtube() -> None:
    assert YouTubeSource.name == youtube.SOURCE_NAME == "youtube"


def test_nvda_registered_to_youtube() -> None:
    """NVDA is the live-smoke target; its registry entry must point
    to `youtube` so the smoke actually exercises this source."""
    from auto_research.ingest.transcripts import registry

    assert registry.REGISTRY["NVDA"] == "youtube"


def test_nvda_has_query_override() -> None:
    """NVDA's query override (`NVIDIA`) must be configured;
    otherwise the ticker symbol could clash with unrelated YouTube
    content on certain quarters."""
    assert "NVDA" in youtube.TICKER_QUERIES
    assert youtube.TICKER_QUERIES["NVDA"].strip()


def test_short_ticker_overrides_present() -> None:
    """Short tickers whose symbols are common English substrings
    (e.g. 'ON' in 'Conference', 'BE' in 'Adobe') MUST have overrides
    — without them, the title gate's substring check false-matches
    unrelated companies' earnings calls. Verified by the SEC-anchored
    universe-wide validation; this test pins the lesson."""
    for ticker in ("ON", "BE"):
        assert ticker in youtube.TICKER_QUERIES, (
            f"{ticker} needs an override — its symbol is a common "
            "English substring and bare-symbol queries false-match."
        )


# ---------- OTel instrumentation (refs #52) ----------


def test_find_audio_url_emits_span_matched_true(
    span_recorder,  # type: ignore[no-untyped-def]
) -> None:
    """A matched search result records matched=True + result_count."""
    src = _make_source(
        search_response={
            "entries": [
                _entry(
                    title="NVIDIA Q1 FY25 Earnings Call",
                    webpage_url="https://youtube.com/watch?v=match",
                ),
            ],
        }
    )
    url = src.find_audio_url("NVDA", 2025, 1)
    assert url == "https://youtube.com/watch?v=match"

    span = span_recorder.one("transcript.find_audio_url")
    assert span.attributes["transcript.ticker"] == "NVDA"
    assert span.attributes["transcript.year"] == 2025
    assert span.attributes["transcript.quarter"] == 1
    assert span.attributes["transcript.matched"] is True
    assert span.attributes["transcript.result_count"] == 1


def test_find_audio_url_emits_span_matched_false(
    span_recorder,  # type: ignore[no-untyped-def]
) -> None:
    src = _make_source(search_response={"entries": []})
    assert src.find_audio_url("NVDA", 2025, 1) is None
    span = span_recorder.one("transcript.find_audio_url")
    assert span.attributes["transcript.matched"] is False
    assert span.attributes["transcript.result_count"] == 0


def test_download_emits_span(
    span_recorder,  # type: ignore[no-untyped-def]
) -> None:
    src = _make_source(audio_bytes=_VALID_AUDIO_BYTES)
    audio = src.download("https://youtube.com/watch?v=fake")
    span = span_recorder.one("transcript.download")
    assert span.attributes["transcript.source_name"] == "youtube"
    assert span.attributes["transcript.bytes"] == len(audio)
    # duration_ms is wall-clock; a fast in-memory fake completes in <1ms,
    # so just assert non-negative.
    assert span.attributes["transcript.duration_ms"] >= 0
