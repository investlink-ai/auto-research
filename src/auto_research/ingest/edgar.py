"""SEC EDGAR public-filings client.

Fetches 10-K, 10-Q, 8-K, S-1, S-3 (and any other configured form types)
for a given CIK and persists raw bytes at
`data/raw/edgar/{cik}/{year}/{accession}.{ext}`. Idempotent on
`(cik, accession_number)` via the append-only Parquet manifest in
`auto_research.ingest.manifest`.

`accepted_datetime` (SEC's `acceptanceDateTime`) is recorded as the
canonical point-in-time stamp — it's the moment EDGAR exposed the
filing to the public and is therefore the earliest timestamp at which
a trading signal could legitimately incorporate it (lag-1 cutoff
applies downstream in Feast — INV-1).

SEC's documented fair-access policy (10 req/s per IP, meaningful
User-Agent required) is honored at every request:

- `SEC_USER_AGENT` is read at construction; missing/blank raises
  `EdgarConfigError` so failures surface at startup, not in opaque
  throttling later.
- All HTTP calls pass through a shared `TokenBucket` rate limiter
  defaulting to 8 req/s (20% margin under the 10 r/s cap).
- Transient errors (timeouts, 429s, 5xx) retry with exponential
  backoff + jitter. Retry-After is honored *via tenacity's wait
  callback* (not double-slept by `_do_request`), and supports both
  delta-seconds and HTTP-date forms (RFC 7231 §7.1.3). The wait is
  clamped to `_MAX_RETRY_AFTER_SECONDS` so a runaway upstream can't
  pin a worker for an unbounded interval.
- Both a sync `EdgarClient` and an `AsyncEdgarClient` are provided;
  they share the same `TokenBucket` instance when both are used in
  the same process (SEC throttles per-IP).

Idempotency is durable: the manifest is the cache key. New rows are
appended in a `try/finally` block — the async path additionally
awaits `gather(return_exceptions=True)` so siblings that complete
after a peer's exception still record their work before the flush.
Cross-process duplication is guarded by `manifest.append`'s
dedup-on-write under the file lock.

Lifecycle: both clients eagerly open their httpx underlying client
in `__init__`. Always close them — either via `client.close()` /
`await client.aclose()`, or by using the (a)sync context-manager
form (`with EdgarClient() as c:` / `async with AsyncEdgarClient()
as c:`). A constructed-but-not-closed client leaks a connection
pool.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from auto_research.ingest import manifest
from auto_research.ingest.rate_limit import TokenBucket, sec_rate_limiter

DEFAULT_FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q", "8-K", "S-1", "S-3")
SOURCE: str = "edgar"

_SUBMISSIONS_BASE = "https://data.sec.gov"
_ARCHIVES_BASE = "https://www.sec.gov"
_ET = ZoneInfo("America/New_York")  # SEC documents naive timestamps as Eastern

_DEFAULT_MAX_ATTEMPTS: Final = 5
_DEFAULT_INITIAL_BACKOFF: Final = 1.0
_DEFAULT_MAX_BACKOFF: Final = 30.0
# Clamp Retry-After so a runaway upstream can't deadlock a worker. SEC's
# observed Retry-After is single-digit seconds; 300s is a generous cap.
_MAX_RETRY_AFTER_SECONDS: Final = 300.0
# Async fan-out concurrency default. Matches the rate limit (8 r/s) so the
# semaphore isn't the bottleneck — see _one() comments. Adjust per-call if
# the downstream HTTP latency profile changes.
_DEFAULT_ASYNC_CONCURRENCY: Final = 8

# EDGAR canonical CIK form is zero-padded to 10 digits.
_CIK_PADDED_PATTERN = re.compile(r"^\d{10}$")

_logger = logging.getLogger(__name__)


class EdgarConfigError(RuntimeError):
    """`SEC_USER_AGENT` env var is missing or blank."""


class EdgarRateLimited(httpx.HTTPStatusError):
    """SEC returned 429. Carries the parsed `Retry-After` hint if present.

    `retry_after` is the seconds-until-retry the server requested, or
    None if absent / unparseable. The retry policy honors it via the
    tenacity wait callback.
    """

    def __init__(self, response: httpx.Response, retry_after: float | None) -> None:
        super().__init__("SEC rate-limited (429)", request=response.request, response=response)
        self.retry_after = retry_after


class EdgarServerError(httpx.HTTPStatusError):
    """SEC returned a 5xx — transient by assumption, eligible for retry."""


class EdgarEmptyResponseError(httpx.HTTPStatusError):
    """200 OK with zero-byte body — almost always a transient CDN /
    propagation issue. Retry-eligible so a brief edge inconsistency
    doesn't poison the manifest with a sha256-of-empty cache hit.
    """


# Network errors that warrant a retry. ReadError/WriteError covers connection
# resets; ConnectError covers DNS / TCP problems; TimeoutException covers
# all of httpx's timeout subclasses.
_TRANSIENT_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)

_RETRYABLE: tuple[type[BaseException], ...] = (
    *_TRANSIENT_NETWORK_ERRORS,
    EdgarRateLimited,
    EdgarServerError,
    EdgarEmptyResponseError,
)


@dataclass(frozen=True, slots=True)
class Filing:
    """One EDGAR filing entry.

    `cik` must be the zero-padded 10-digit canonical form. We validate
    in `__post_init__` rather than trusting callers — a Filing built
    with unpadded CIK would silently produce a divergent on-disk
    layout (`data/raw/edgar/1045810/...` vs the orchestrator's
    `.../0001045810/...`).
    """

    cik: str
    accession_number: str
    form_type: str
    primary_document: str
    accepted_datetime: datetime

    def __post_init__(self) -> None:
        if not _CIK_PADDED_PATTERN.match(self.cik):
            raise ValueError(
                f"cik must be zero-padded 10 digits (canonical EDGAR form); got {self.cik!r}"
            )
        if not self.accession_number.strip():
            raise ValueError("accession_number must be non-empty")
        if not self.primary_document.strip():
            raise ValueError("primary_document must be non-empty")


@dataclass(frozen=True, slots=True)
class FetchResult:
    cik: str
    accession_number: str
    form_type: str
    accepted_datetime: datetime
    path: Path | None
    content_sha256: str | None
    cache_hit: bool


def _resolve_user_agent(user_agent: str | None) -> str:
    raw = user_agent if user_agent is not None else os.environ.get("SEC_USER_AGENT", "")
    cleaned = raw.strip()
    if not cleaned:
        raise EdgarConfigError(
            "SEC requires a meaningful User-Agent header for data.sec.gov / "
            "www.sec.gov requests. Set SEC_USER_AGENT in the environment "
            "(e.g., 'Your Name your@email.com')."
        )
    return cleaned


def _default_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/html,application/xhtml+xml,*/*;q=0.8",
    }


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value.

    RFC 7231 §7.1.3 permits two forms: delta-seconds (e.g., `120`) or
    HTTP-date (e.g., `Wed, 21 Oct 2026 07:28:00 GMT`). Returns the
    seconds to wait, or None if the header is absent or unparseable.
    Result is always clamped to `_MAX_RETRY_AFTER_SECONDS` before
    being returned.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    # Try delta-seconds first.
    try:
        seconds = float(stripped)
    except ValueError:
        # Fall back to HTTP-date.
        try:
            target = parsedate_to_datetime(stripped)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        seconds = (target - datetime.now(UTC)).total_seconds()
    return min(max(0.0, seconds), _MAX_RETRY_AFTER_SECONDS)


def _classify_response(response: httpx.Response) -> None:
    """Raise a typed retryable error on 429 / 5xx / empty 200; leave others alone.

    Empty 200s are reclassified because a sha256-of-empty manifest row
    would otherwise become a permanent cache hit on what's almost
    always a transient CDN/propagation gap. Triggered for both
    `data.sec.gov` (submissions JSON) and `www.sec.gov` (Archives) —
    neither should ever legitimately serve a 0-byte body for the
    endpoints this client hits.
    """
    if response.status_code == 429:
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
        raise EdgarRateLimited(response, retry_after)
    if 500 <= response.status_code < 600:
        raise EdgarServerError(
            f"SEC returned {response.status_code}",
            request=response.request,
            response=response,
        )
    if response.status_code == 200 and not response.content:
        raise EdgarEmptyResponseError(
            f"SEC returned 200 with empty body for {response.request.url}",
            request=response.request,
            response=response,
        )


def _retry_wait(
    *,
    initial: float = _DEFAULT_INITIAL_BACKOFF,
    max_wait: float = _DEFAULT_MAX_BACKOFF,
) -> Any:
    """Tenacity wait callback that honors EdgarRateLimited.retry_after.

    For a 429 with a server-provided Retry-After, waits at least that
    long (clamped to `_MAX_RETRY_AFTER_SECONDS`). For any other
    retryable failure, falls back to exponential jitter. This
    eliminates the double-sleep that would otherwise occur if
    `_do_request` slept for Retry-After and then tenacity slept again
    for its exponential backoff.
    """
    fallback = wait_exponential_jitter(initial=initial, max=max_wait)

    def wait(retry_state: RetryCallState) -> float:
        base = float(fallback(retry_state))
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        if isinstance(exc, EdgarRateLimited) and exc.retry_after is not None:
            return max(exc.retry_after, base)
        return base

    return wait


def _atomic_write_bytes(dest: Path, content: bytes) -> None:
    """Write bytes to `dest` atomically and durably.

    Mirrors `manifest.append`'s discipline: tmp file + fsync of both
    the file and the parent directory before the rename, so a power
    loss can't leave a 0-byte canonical file when the manifest row
    is already durably committed. Tmp is removed on any failure
    between `write_bytes` and `os.replace` to avoid leaking hidden
    dotfiles into the destination directory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Tmp name namespaces by both PID and thread id so multiple worker
    # threads under `asyncio.to_thread` writing to the same dest (e.g., a
    # future deliberate-retry pattern) can't collide on the tmp filename.
    tmp = dest.parent / f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        tmp.write_bytes(content)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, dest)
    except BaseException:
        # Cleanup must NOT mask the original exception. unlink can itself
        # raise (PermissionError on read-only fs, EIO on a dying disk) —
        # swallow OSError from cleanup and re-raise the original failure.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    dir_fd = os.open(dest.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class EdgarClient:
    """Sync HTTP client with rate limiting + retries.

    The underlying `httpx.Client` is opened eagerly in `__init__`.
    Always close it — either explicitly via `.close()` or by using
    the context manager form `with EdgarClient() as c: ...`.
    """

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout: float = 30.0,
        rate_limiter: TokenBucket | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        ua = _resolve_user_agent(user_agent)
        self.rate_limiter = rate_limiter or sec_rate_limiter()
        self._max_attempts = max_attempts
        self._client = httpx.Client(
            headers=_default_headers(ua),
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _do_request(self, url: str) -> httpx.Response:
        """One rate-limited GET. Raises retryable errors on 429/5xx.

        Does NOT sleep for Retry-After here — tenacity's wait callback
        owns all backoff timing. This avoids the previous double-sleep
        bug where `_do_request` slept for Retry-After AND tenacity then
        slept again for exponential jitter.
        """
        self.rate_limiter.wait()
        resp = self._client.get(url)
        _classify_response(resp)
        return resp

    def _get(self, url: str) -> httpx.Response:
        """Rate-limited GET with retries on transient failures."""
        for attempt in Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=_retry_wait(),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                return self._do_request(url)
        raise RuntimeError("unreachable: tenacity always returns or raises")

    def list_recent_filings(
        self,
        cik: str | int,
        *,
        form_types: Iterable[str],
    ) -> list[Filing]:
        cik_padded = _pad_cik(cik)
        resp = self._get(f"{_SUBMISSIONS_BASE}/submissions/CIK{cik_padded}.json")
        resp.raise_for_status()
        body = resp.json()
        recent = body["filings"]["recent"]
        return _parse_recent(cik_padded, recent, form_types)

    def fetch_filing(self, filing: Filing, *, raw_root: Path) -> tuple[Path, str, bytes]:
        """Download the filing's primary document. Atomic + durable on disk.

        Empty-body 200s are retried automatically by `_get` (see
        `_classify_response`); by the time this function sees the
        response, the body is non-empty.
        """
        url = _archives_url(filing)
        resp = self._get(url)
        resp.raise_for_status()
        content = resp.content
        sha = hashlib.sha256(content).hexdigest()
        dest = _destination_path(filing, raw_root)
        _atomic_write_bytes(dest, content)
        return dest, sha, content


class AsyncEdgarClient:
    """Async counterpart to `EdgarClient`. Same rate limit, same retries.

    Useful for fan-out across many CIKs. The rate limiter is shared
    across coroutines via a `threading.Lock` internally, so the 8 r/s
    budget is global to the client, not per-coroutine.

    The underlying `httpx.AsyncClient` is opened eagerly in `__init__`.
    Always close it — either explicitly via `await client.aclose()` or
    by using the async context-manager form
    `async with AsyncEdgarClient() as c: ...`. A constructed-but-not-
    closed client leaks an asyncio connection pool.
    """

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout: float = 30.0,
        rate_limiter: TokenBucket | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        ua = _resolve_user_agent(user_agent)
        self.rate_limiter = rate_limiter or sec_rate_limiter()
        self._max_attempts = max_attempts
        self._client = httpx.AsyncClient(
            headers=_default_headers(ua),
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncEdgarClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _do_request(self, url: str) -> httpx.Response:
        await self.rate_limiter.wait_async()
        resp = await self._client.get(url)
        _classify_response(resp)
        return resp

    async def _get(self, url: str) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=_retry_wait(),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                return await self._do_request(url)
        raise RuntimeError("unreachable: tenacity always returns or raises")

    async def list_recent_filings(
        self,
        cik: str | int,
        *,
        form_types: Iterable[str],
    ) -> list[Filing]:
        cik_padded = _pad_cik(cik)
        resp = await self._get(f"{_SUBMISSIONS_BASE}/submissions/CIK{cik_padded}.json")
        resp.raise_for_status()
        body = resp.json()
        return _parse_recent(cik_padded, body["filings"]["recent"], form_types)

    async def fetch_filing(self, filing: Filing, *, raw_root: Path) -> tuple[Path, str, bytes]:
        """Async counterpart to sync EdgarClient.fetch_filing.

        The atomic write + fsync sequence is sync I/O, so we dispatch
        it to a thread (`asyncio.to_thread`) to keep the event loop
        responsive — otherwise a multi-MB filing's fsync would stall
        every other coroutine on the same loop. Empty-body 200s are
        retried automatically by `_get` (see `_classify_response`).
        """
        url = _archives_url(filing)
        resp = await self._get(url)
        resp.raise_for_status()
        content = resp.content
        sha = hashlib.sha256(content).hexdigest()
        dest = _destination_path(filing, raw_root)
        await asyncio.to_thread(_atomic_write_bytes, dest, content)
        return dest, sha, content


def fetch_filings_for_cik(
    cik: str | int,
    *,
    form_types: Iterable[str] = DEFAULT_FORM_TYPES,
    raw_root: Path,
    manifest_path: Path,
    client: EdgarClient | None = None,
) -> list[FetchResult]:
    """Fetch every requested filing for `cik`; record results in the manifest.

    Idempotency: a snapshot of already-recorded `(source, doc_id)` is
    taken once at entry, so the per-filing check is O(1) regardless
    of manifest size. New rows are appended to the manifest in a
    `try/finally` block so a mid-loop crash still persists what
    succeeded. `manifest.append` additionally dedups on `(source,
    doc_id)` inside the file lock, so two concurrent processes that
    both miss the snapshot can't double-insert.

    `raw_root` is the project's `data/raw/` root; output is nested
    under `raw_root/edgar/{cik}/{year}/{accession}.{ext}`.
    """
    owns_client = client is None
    if owns_client:
        client = EdgarClient()
    assert client is not None
    try:
        filings = client.list_recent_filings(cik, form_types=form_types)
        already = manifest.existing_doc_ids(manifest_path, source=SOURCE)
        results: list[FetchResult] = []
        new_rows: list[dict[str, object]] = []
        seen: set[str] = set()
        try:
            for filing in filings:
                if filing.accession_number in seen:
                    continue
                seen.add(filing.accession_number)
                if filing.accession_number in already:
                    results.append(_cache_hit(filing))
                    continue
                path, sha, _ = client.fetch_filing(filing, raw_root=raw_root / SOURCE)
                results.append(
                    FetchResult(
                        cik=filing.cik,
                        accession_number=filing.accession_number,
                        form_type=filing.form_type,
                        accepted_datetime=filing.accepted_datetime,
                        path=path,
                        content_sha256=sha,
                        cache_hit=False,
                    )
                )
                new_rows.append(_manifest_row(filing, path, sha, status="ok"))
        finally:
            if new_rows:
                manifest.append(manifest_path, new_rows)
        return results
    finally:
        if owns_client:
            client.close()


async def afetch_filings_for_cik(
    cik: str | int,
    *,
    form_types: Iterable[str] = DEFAULT_FORM_TYPES,
    raw_root: Path,
    manifest_path: Path,
    client: AsyncEdgarClient | None = None,
    concurrency: int = _DEFAULT_ASYNC_CONCURRENCY,
) -> list[FetchResult]:
    """Async fan-out variant. Fetches up to `concurrency` filings in parallel.

    Uses `asyncio.gather(return_exceptions=True)` so the gather call
    only returns after EVERY child coroutine has either appended its
    row to `new_rows` or raised — which is the load-bearing property
    that lets the `finally` flush capture all successful work. The
    `return_exceptions` name suggests "swallow errors", but its real
    value here is the await-all-children semantics; failures are
    re-raised as a `BaseExceptionGroup` so callers see every failure
    mode, not just the first.

    Manifest append and the per-filing atomic-write are sync I/O,
    so both are dispatched via `asyncio.to_thread` to keep the event
    loop responsive during the (potentially hundreds of ms of)
    `fcntl.flock` + `pq.write_table` + `os.fsync` work.

    The shared rate limiter still caps total throughput at SEC's
    8 r/s; the semaphore caps simultaneously in-flight downloads.
    `concurrency` defaults to match the rate limit so the semaphore
    isn't the bottleneck for typical HTTP latencies.

    `asyncio.CancelledError` from any child or from outer cancellation
    is re-raised immediately (preserving structured-concurrency
    semantics), bypassing the group wrap.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    owns_client = client is None
    if owns_client:
        client = AsyncEdgarClient()
    assert client is not None
    try:
        filings = await client.list_recent_filings(cik, form_types=form_types)
        already = manifest.existing_doc_ids(manifest_path, source=SOURCE)
        seen: set[str] = set()
        deduped: list[Filing] = []
        cached_results: list[FetchResult] = []
        for filing in filings:
            if filing.accession_number in seen:
                continue
            seen.add(filing.accession_number)
            if filing.accession_number in already:
                cached_results.append(_cache_hit(filing))
            else:
                deduped.append(filing)

        sem = asyncio.Semaphore(concurrency)
        new_rows: list[dict[str, object]] = []
        fetched_results: list[FetchResult] = []

        async def _one(filing: Filing) -> None:
            # Acquire the rate-limiter slot OUTSIDE the semaphore so the
            # semaphore counts only active HTTP requests, not coroutines
            # parked inside the bucket waiting for a token. With sem ==
            # rate, this matters less; with sem < rate or under contention
            # it preserves the budgeted throughput.
            assert client is not None  # narrowed for the closure
            async with sem:
                path, sha, _ = await client.fetch_filing(filing, raw_root=raw_root / SOURCE)
            fetched_results.append(
                FetchResult(
                    cik=filing.cik,
                    accession_number=filing.accession_number,
                    form_type=filing.form_type,
                    accepted_datetime=filing.accepted_datetime,
                    path=path,
                    content_sha256=sha,
                    cache_hit=False,
                )
            )
            new_rows.append(_manifest_row(filing, path, sha, status="ok"))

        exceptions: list[BaseException] = []
        try:
            results = await asyncio.gather(
                *(_one(f) for f in deduped), return_exceptions=True
            )
            exceptions = [r for r in results if isinstance(r, BaseException)]
        finally:
            if new_rows:
                # Catch the flush exception so it joins the collected per-filing
                # failures instead of replacing them. Otherwise a disk-full at
                # flush time would mask every EdgarRateLimited / EdgarServerError
                # gather already captured.
                try:
                    await asyncio.to_thread(manifest.append, manifest_path, new_rows)
                except BaseException as flush_exc:
                    exceptions.append(flush_exc)
        # Honor structured cancellation: surface CancelledError directly so
        # outer task-group / wait_for semantics stay intact, but attach
        # suppressed sibling failures as notes (PEP 678) instead of dropping
        # them silently.
        cancelled = next(
            (e for e in exceptions if isinstance(e, asyncio.CancelledError)),
            None,
        )
        if cancelled is not None:
            for other in exceptions:
                if other is cancelled:
                    continue
                cancelled.add_note(f"suppressed: {type(other).__name__}: {other}")
            raise cancelled
        if exceptions:
            raise BaseExceptionGroup("EDGAR fetches failed", exceptions)

        return [*cached_results, *fetched_results]
    finally:
        if owns_client:
            await client.aclose()


# ---------- helpers ----------


def _archives_url(filing: Filing) -> str:
    accession_nodash = filing.accession_number.replace("-", "")
    cik_unpadded = str(int(filing.cik))  # Archives URL uses unpadded CIK
    return (
        f"{_ARCHIVES_BASE}/Archives/edgar/data/{cik_unpadded}"
        f"/{accession_nodash}/{filing.primary_document}"
    )


def _destination_path(filing: Filing, raw_root: Path) -> Path:
    year = filing.accepted_datetime.astimezone(UTC).year
    ext = Path(filing.primary_document).suffix or ".bin"
    return raw_root / filing.cik / str(year) / f"{filing.accession_number}{ext}"


def _cache_hit(filing: Filing) -> FetchResult:
    return FetchResult(
        cik=filing.cik,
        accession_number=filing.accession_number,
        form_type=filing.form_type,
        accepted_datetime=filing.accepted_datetime,
        path=None,
        content_sha256=None,
        cache_hit=True,
    )


def _manifest_row(filing: Filing, path: Path, sha: str, *, status: str) -> dict[str, object]:
    return {
        "source": SOURCE,
        "entity_id": filing.cik,
        "doc_id": filing.accession_number,
        "form_type": filing.form_type,
        "event_datetime": filing.accepted_datetime,
        "fetched_at": datetime.now(UTC),
        "content_sha256": sha,
        "path": str(path),
        "status": status,
    }


def _parse_recent(
    cik_padded: str,
    recent: dict[str, list[Any]],
    form_types: Iterable[str],
) -> list[Filing]:
    """Zip parallel arrays with `strict=True` so misalignment fails loudly.

    Skips entries with empty/blank `accessionNumber` or `primaryDocument`
    — both would silently produce malformed Filings (Directory-index
    URLs, empty doc_id collisions in the dedup key, etc.).
    """
    wanted = set(form_types)
    out: list[Filing] = []
    for form, accession, primary, accepted in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["acceptanceDateTime"],
        strict=True,
    ):
        if form not in wanted:
            continue
        if not (isinstance(accession, str) and accession.strip()):
            _logger.warning(
                "skipping row with blank accessionNumber for cik=%s form=%s",
                cik_padded,
                form,
            )
            continue
        if not (isinstance(primary, str) and primary.strip()):
            _logger.warning(
                "skipping row with blank primaryDocument for cik=%s accession=%s",
                cik_padded,
                accession,
            )
            continue
        if not (isinstance(accepted, str) and accepted.strip()):
            _logger.warning(
                "skipping row with blank acceptanceDateTime for cik=%s accession=%s",
                cik_padded,
                accession,
            )
            continue
        out.append(
            Filing(
                cik=cik_padded,
                accession_number=accession,
                form_type=form,
                primary_document=primary,
                accepted_datetime=_parse_acceptance(accepted),
            )
        )
    return out


def _pad_cik(cik: str | int) -> str:
    return f"{int(cik):010d}"


def _parse_acceptance(value: str) -> datetime:
    """Parse SEC's `acceptanceDateTime`.

    EDGAR returns Z-suffixed UTC in practice. The spec also permits
    naive timestamps (no `Z`, no offset), which SEC documents as
    Eastern. We honor that: naive → America/New_York → UTC. Refuses
    to silently misclassify after-hours filings into the wrong
    trading day (INV-1 adjacent).
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    return dt.astimezone(UTC)


__all__ = [
    "DEFAULT_FORM_TYPES",
    "SOURCE",
    "AsyncEdgarClient",
    "EdgarClient",
    "EdgarConfigError",
    "EdgarEmptyResponseError",
    "EdgarRateLimited",
    "EdgarServerError",
    "FetchResult",
    "Filing",
    "afetch_filings_for_cik",
    "fetch_filings_for_cik",
]
