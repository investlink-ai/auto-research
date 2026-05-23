"""Unit tests for `auto_research.ingest._http` shared helpers.

These tests previously lived in `test_edgar.py` (when the helpers were
private to that module). Moved here when the helpers were extracted to
`_http.py` so EDGAR + FMP (and any future ingest source) share one
correct implementation. No back-compat re-exports kept on the EDGAR
module — per the user's stated "no scattered duplication" policy, a
single source of truth wins.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from auto_research.ingest import _http


def test_parse_retry_after_delta_seconds() -> None:
    assert _http.parse_retry_after("30") == 30.0
    assert _http.parse_retry_after("  120  ") == 120.0
    assert _http.parse_retry_after(None) is None
    assert _http.parse_retry_after("") is None


def test_parse_retry_after_http_date() -> None:
    """RFC 7231 HTTP-date form parses to a positive delta-seconds value."""
    parsed = _http.parse_retry_after("Thu, 01 Jan 2099 00:00:00 GMT")
    assert parsed is not None
    assert parsed > 0
    assert parsed <= _http.MAX_RETRY_AFTER_SECONDS


def test_parse_retry_after_clamps_to_max() -> None:
    """A runaway server value clamps to MAX_RETRY_AFTER_SECONDS."""
    assert _http.parse_retry_after("999999") == _http.MAX_RETRY_AFTER_SECONDS


def test_parse_retry_after_garbage_returns_none() -> None:
    assert _http.parse_retry_after("not-a-thing") is None


def test_atomic_write_bytes_cleans_tmp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fsync/replace error must not leak the hidden tmp file."""
    dest = tmp_path / "subdir" / "x.htm"

    def boom_replace(src: object, dst: object, *args: object, **kwargs: object) -> None:
        raise OSError("simulated disk-full")

    monkeypatch.setattr(os, "replace", boom_replace)
    with pytest.raises(OSError, match="simulated disk-full"):
        _http.atomic_write_bytes(dest, b"hello")

    leftovers = list(dest.parent.glob(".*.tmp"))
    assert leftovers == [], f"tmp file leaked: {leftovers}"


def test_atomic_write_bytes_happy_path(tmp_path: Path) -> None:
    dest = tmp_path / "subdir" / "y.htm"
    _http.atomic_write_bytes(dest, b"payload")
    assert dest.read_bytes() == b"payload"
    # No tmp file leaks left in dest.parent.
    leftovers = list(dest.parent.glob(".*.tmp"))
    assert leftovers == []


def test_default_headers_includes_accept_encoding() -> None:
    h = _http.default_headers(user_agent="Test ua@example.com")
    assert h["User-Agent"] == "Test ua@example.com"
    assert "gzip" in h["Accept-Encoding"]


def test_default_headers_omits_user_agent_when_none() -> None:
    """Sources that auth via query param (FMP) don't need a UA."""
    h = _http.default_headers()
    assert "User-Agent" not in h


def test_retryable_exceptions_includes_transients_and_source_classes() -> None:
    class _RL(_http.RateLimited):
        pass

    class _SE(_http.ServerError):
        pass

    class _ER(_http.EmptyResponseError):
        pass

    retryable = _http.retryable_exceptions(
        rate_limited=_RL, server_error=_SE, empty_response=_ER
    )
    assert _RL in retryable
    assert _SE in retryable
    assert _ER in retryable
    # And the transient httpx errors.
    for cls in _http.TRANSIENT_NETWORK_ERRORS:
        assert cls in retryable
