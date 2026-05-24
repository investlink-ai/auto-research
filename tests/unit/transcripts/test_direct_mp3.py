"""Unit tests for the `direct_mp3` source."""

from __future__ import annotations

import httpx
import pytest

from auto_research.ingest._http import RateLimited
from auto_research.ingest.rate_limit import TokenBucket
from auto_research.ingest.transcripts._base import TranscriptConfigError
from auto_research.ingest.transcripts.sources import direct_mp3


def _fast_limiter() -> TokenBucket:
    """Effectively-no-op rate limiter for tests."""
    return TokenBucket(rate=1000.0, capacity=1000.0)


def _mock_transport(handler: object) -> httpx.MockTransport:
    return httpx.MockTransport(handler)  # type: ignore[arg-type]


# ---------- find_audio_url ----------


def test_find_audio_url_returns_none_for_unregistered_ticker() -> None:
    src = direct_mp3.DirectMp3Source(rate_limiter=_fast_limiter())
    try:
        assert src.find_audio_url("NEVER", 2024, 1) is None
    finally:
        src.close()


def test_find_audio_url_formats_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        direct_mp3.TICKER_URL_TEMPLATES,
        "ACME",
        "https://investor.acme.com/audio/{year}Q{quarter}.mp3",
    )
    src = direct_mp3.DirectMp3Source(rate_limiter=_fast_limiter())
    try:
        assert (
            src.find_audio_url("ACME", 2024, 3)
            == "https://investor.acme.com/audio/2024Q3.mp3"
        )
        # Case-insensitive ticker lookup.
        assert (
            src.find_audio_url("acme", 2024, 3)
            == "https://investor.acme.com/audio/2024Q3.mp3"
        )
    finally:
        src.close()


def test_find_audio_url_rejects_unknown_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A template with an unsupported {placeholder} should raise loudly
    at lookup time — not silently leave the brace in the URL."""
    monkeypatch.setitem(
        direct_mp3.TICKER_URL_TEMPLATES,
        "BADTPL",
        "https://example.com/{ticker}/{year}Q{quarter}.mp3",
    )
    src = direct_mp3.DirectMp3Source(rate_limiter=_fast_limiter())
    try:
        with pytest.raises(TranscriptConfigError, match="placeholder"):
            src.find_audio_url("BADTPL", 2024, 1)
    finally:
        src.close()


# ---------- download ----------


def test_download_returns_bytes_for_2xx() -> None:
    payload = b"\xff\xfb" + b"\x00" * 100  # MP3 header sentinel + padding

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, headers={"content-type": "audio/mpeg"})

    src = direct_mp3.DirectMp3Source(
        rate_limiter=_fast_limiter(),
        transport=_mock_transport(handler),
    )
    try:
        assert src.download("https://example.com/a.mp3") == payload
    finally:
        src.close()


def test_download_retries_on_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Suppress real tenacity backoff sleeps for speed.
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="bad gateway")
        return httpx.Response(200, content=b"ok")

    src = direct_mp3.DirectMp3Source(
        rate_limiter=_fast_limiter(),
        max_attempts=5,
        transport=_mock_transport(handler),
    )
    try:
        assert src.download("https://example.com/a.mp3") == b"ok"
        assert calls["n"] == 3
    finally:
        src.close()


def test_download_honors_429_via_shared_retry_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, content=b"ok")

    src = direct_mp3.DirectMp3Source(
        rate_limiter=_fast_limiter(),
        max_attempts=3,
        transport=_mock_transport(handler),
    )
    try:
        assert src.download("https://example.com/a.mp3") == b"ok"
        assert calls["n"] == 2
    finally:
        src.close()


def test_download_raises_on_4xx() -> None:
    """4xx (other than 429 which is handled as retryable) raises
    `httpx.HTTPStatusError` — `_http.classify_response` doesn't
    catch it; `resp.raise_for_status` does."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    src = direct_mp3.DirectMp3Source(
        rate_limiter=_fast_limiter(),
        max_attempts=1,
        transport=_mock_transport(handler),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            src.download("https://example.com/missing.mp3")
    finally:
        src.close()


def test_download_classifies_429_as_rate_limited() -> None:
    """The classifier should produce the shared `RateLimited` type so
    the retry callback can read `retry_after`."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"})

    src = direct_mp3.DirectMp3Source(
        rate_limiter=_fast_limiter(),
        max_attempts=1,  # don't retry past the first attempt
        transport=_mock_transport(handler),
    )
    try:
        with pytest.raises(RateLimited) as excinfo:
            src.download("https://example.com/a.mp3")
        assert excinfo.value.retry_after == 1.0
    finally:
        src.close()


# ---------- source identity / Protocol shape ----------


def test_source_has_name_attribute() -> None:
    assert direct_mp3.DirectMp3Source.name == direct_mp3.SOURCE_NAME == "direct_mp3"


def test_source_in_known_sources() -> None:
    """Registry sanity-check needs the source name in KNOWN_SOURCES."""
    from auto_research.ingest.transcripts import registry

    assert direct_mp3.SOURCE_NAME in registry.KNOWN_SOURCES


# ---------- OTel instrumentation (refs #52) ----------


def test_download_emits_span(span_recorder) -> None:  # type: ignore[no-untyped-def]
    payload = b"\xff\xfb" + b"\x00" * 256

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, headers={"content-type": "audio/mpeg"})

    src = direct_mp3.DirectMp3Source(
        rate_limiter=_fast_limiter(),
        transport=_mock_transport(handler),
    )
    try:
        result = src.download("https://example.com/audio.mp3")
    finally:
        src.close()
    assert result == payload

    span = span_recorder.one("transcript.download")
    assert span.attributes["transcript.source_name"] == "direct_mp3"
    assert span.attributes["transcript.bytes"] == len(payload)
    assert span.attributes["transcript.duration_ms"] >= 0
