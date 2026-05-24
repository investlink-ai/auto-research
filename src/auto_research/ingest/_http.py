"""Shared HTTP retry / classification / atomic-write helpers for ingest modules.

Every ingest source shares the same HTTP discipline:

- rate-limited via `auto_research.ingest.rate_limit.TokenBucket`
- 429 / 5xx / empty-200 reclassified as typed retryable errors
- tenacity-driven exponential backoff with jitter, honoring Retry-After
- atomic write to disk via tmp + fsync + rename

Per-source modules subclass `RateLimited`, `ServerError`, `EmptyResponseError`
for catch-site ergonomics (`except EdgarRateLimited:` reads better than a
generic `RateLimited` at the call site). The shared `classify_response`
and `make_retry_wait` helpers operate on the BASE classes — they work
across sources without parameterization on the wait callback, and only
need the per-source concrete classes plumbed into `classify_response`
to construct the right typed exception.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final

import httpx
from tenacity import RetryCallState, wait_exponential_jitter

from auto_research._io import atomic_write_bytes
from auto_research._transport import TRANSIENT_NETWORK_ERRORS

# Tunables. Sources may pass overrides (e.g., a slower upstream might
# want a higher max_attempts; a faster one a tighter backoff).
DEFAULT_MAX_ATTEMPTS: Final = 5
DEFAULT_INITIAL_BACKOFF: Final = 1.0
DEFAULT_MAX_BACKOFF: Final = 30.0
# Clamp Retry-After so a runaway upstream can't deadlock a worker. Most
# real-world Retry-After values are single-digit seconds; 300s is a
# generous cap.
MAX_RETRY_AFTER_SECONDS: Final = 300.0


class RateLimited(httpx.HTTPStatusError):
    """429 response. Sources subclass for catch-site ergonomics.

    `retry_after` is the parsed seconds-until-retry the server requested,
    or None if absent/unparseable. The shared retry-wait callback honors
    it via `isinstance(exc, RateLimited)` so subclasses don't need to
    re-declare the field.
    """

    def __init__(
        self,
        message: str,
        *,
        response: httpx.Response,
        retry_after: float | None,
    ) -> None:
        super().__init__(message, request=response.request, response=response)
        self.retry_after = retry_after


class ServerError(httpx.HTTPStatusError):
    """5xx response — transient by assumption, eligible for retry."""


class EmptyResponseError(httpx.HTTPStatusError):
    """200 OK with zero-byte body — almost always a transient CDN /
    propagation issue. Retry-eligible so a sha256-of-empty manifest row
    doesn't become a permanent cache hit on what's actually a transient
    edge inconsistency.
    """


def retryable_exceptions(
    *,
    rate_limited: type[RateLimited],
    server_error: type[ServerError],
    empty_response: type[EmptyResponseError],
) -> tuple[type[BaseException], ...]:
    """Build the tenacity `retry_if_exception_type` tuple for a source."""
    return (*TRANSIENT_NETWORK_ERRORS, rate_limited, server_error, empty_response)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value.

    RFC 7231 §7.1.3 permits two forms: delta-seconds (e.g., `120`) or
    HTTP-date (e.g., `Wed, 21 Oct 2026 07:28:00 GMT`). Returns the
    seconds to wait, or None if the header is absent or unparseable.
    Result is always clamped to `MAX_RETRY_AFTER_SECONDS`.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            target = parsedate_to_datetime(stripped)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        seconds = (target - datetime.now(UTC)).total_seconds()
    return min(max(0.0, seconds), MAX_RETRY_AFTER_SECONDS)


def classify_response(
    response: httpx.Response,
    *,
    rate_limited: type[RateLimited],
    server_error: type[ServerError],
    empty_response: type[EmptyResponseError],
    source_label: str,
) -> None:
    """Raise the per-source typed retryable on 429 / 5xx / empty-200.

    `source_label` is a short human-readable name (e.g., `"SEC"`, `"FMP"`)
    embedded in the exception message so tracebacks identify which
    upstream service failed without inspecting `response.request.url`.
    Keyword-required (no default) — forgetting it would produce
    generic "upstream X-on-Y" messages that obscure which source
    raised, complicating multi-source incident triage.

    `response.request.url` is touched only inside the error branches —
    a synthetic `httpx.Response` constructed without a request (a future
    unit-test pattern) doesn't fail the no-op fast path.
    """
    status = response.status_code
    if status == 429:
        retry_after = parse_retry_after(response.headers.get("Retry-After"))
        raise rate_limited(
            f"{source_label} rate-limited (429) on {response.request.url}",
            response=response,
            retry_after=retry_after,
        )
    if 500 <= status < 600:
        raise server_error(
            f"{source_label} returned {status} on {response.request.url}",
            request=response.request,
            response=response,
        )
    if status == 200 and not response.content:
        raise empty_response(
            f"{source_label} returned 200 with empty body for {response.request.url}",
            request=response.request,
            response=response,
        )


def make_retry_wait(
    *,
    initial: float = DEFAULT_INITIAL_BACKOFF,
    max_wait: float = DEFAULT_MAX_BACKOFF,
) -> Any:
    """Tenacity wait callback that honors `RateLimited.retry_after`.

    For a 429 with a server-provided Retry-After, waits at least that
    long (already clamped at parse time). For any other retryable
    failure, falls back to exponential jitter. This avoids the
    double-sleep that would otherwise occur if the client slept for
    Retry-After explicitly AND then tenacity slept again for its
    exponential backoff.
    """
    fallback = wait_exponential_jitter(initial=initial, max=max_wait)

    def wait(state: RetryCallState) -> float:
        base = float(fallback(state))
        outcome = state.outcome
        exc = outcome.exception() if outcome is not None else None
        if isinstance(exc, RateLimited) and exc.retry_after is not None:
            return max(exc.retry_after, base)
        return base

    return wait


def default_headers(
    *,
    user_agent: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the default HTTP headers for an ingest client.

    `user_agent` is optional — clients that authenticate via query
    param (e.g., FMP's `apikey=`) don't have a fair-access UA policy
    to honor. Sources that DO have one (SEC) should pass their
    resolved UA string. `extra` lets a source add source-specific
    headers (e.g., a custom Accept type).

    Raises `ValueError` if both `user_agent` and `extra` carry a
    `User-Agent` — the silent precedence would otherwise let an
    `extra` entry clobber a validated UA without warning. Be
    explicit: pass UA via `user_agent`, route other headers via
    `extra`.
    """
    if extra and user_agent and any(k.lower() == "user-agent" for k in extra):
        raise ValueError(
            "User-Agent supplied via both `user_agent` and `extra`. "
            "Pass it through `user_agent` only; route other headers via `extra`."
        )
    headers: dict[str, str] = {
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/html,application/xhtml+xml,*/*;q=0.8",
    }
    if user_agent:
        headers["User-Agent"] = user_agent
    if extra:
        headers.update(extra)
    return headers


__all__ = [
    "DEFAULT_INITIAL_BACKOFF",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_BACKOFF",
    "MAX_RETRY_AFTER_SECONDS",
    "TRANSIENT_NETWORK_ERRORS",
    "EmptyResponseError",
    "RateLimited",
    "ServerError",
    "atomic_write_bytes",
    "classify_response",
    "default_headers",
    "make_retry_wait",
    "parse_retry_after",
    "retryable_exceptions",
]
