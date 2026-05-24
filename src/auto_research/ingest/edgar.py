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
  clamped to `_http.MAX_RETRY_AFTER_SECONDS` so a runaway upstream
  can't pin a worker for an unbounded interval.
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
import hashlib
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import httpx
from opentelemetry import trace
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from auto_research.ingest import _http, manifest
from auto_research.ingest.rate_limit import TokenBucket, sec_rate_limiter

_tracer = trace.get_tracer(__name__)

DEFAULT_FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q", "8-K", "S-1", "S-3")
SOURCE: str = "edgar"
SOURCE_LABEL: Final = "SEC"

_SUBMISSIONS_BASE = "https://data.sec.gov"
_ARCHIVES_BASE = "https://www.sec.gov"
_ET = ZoneInfo("America/New_York")  # SEC documents naive timestamps as Eastern

# Async fan-out concurrency default. Matches the rate limit (8 r/s) so the
# semaphore isn't the bottleneck — see _one() comments. Adjust per-call if
# the downstream HTTP latency profile changes.
_DEFAULT_ASYNC_CONCURRENCY: Final = 8

# EDGAR canonical CIK form is zero-padded to 10 digits.
_CIK_PADDED_PATTERN = re.compile(r"^\d{10}$")

_logger = logging.getLogger(__name__)


class EdgarConfigError(RuntimeError):
    """`SEC_USER_AGENT` env var is missing or blank."""


class EdgarRateLimited(_http.RateLimited):
    """SEC returned 429. See `_http.RateLimited` for the carried `retry_after`."""


class EdgarServerError(_http.ServerError):
    """SEC returned a 5xx — transient by assumption, eligible for retry."""


class EdgarEmptyResponseError(_http.EmptyResponseError):
    """SEC returned 200 with empty body — retried per `_http.EmptyResponseError`."""


_RETRYABLE = _http.retryable_exceptions(
    rate_limited=EdgarRateLimited,
    server_error=EdgarServerError,
    empty_response=EdgarEmptyResponseError,
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


def _classify_response(response: httpx.Response) -> None:
    """SEC-specific wrapper around `_http.classify_response`."""
    _http.classify_response(
        response,
        rate_limited=EdgarRateLimited,
        server_error=EdgarServerError,
        empty_response=EdgarEmptyResponseError,
        source_label=SOURCE_LABEL,
    )


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
        max_attempts: int = _http.DEFAULT_MAX_ATTEMPTS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        ua = _resolve_user_agent(user_agent)
        self.rate_limiter = rate_limiter or sec_rate_limiter()
        self._max_attempts = max_attempts
        self._client = httpx.Client(
            headers=_http.default_headers(user_agent=ua),
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
            wait=_http.make_retry_wait(),
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
        _http.atomic_write_bytes(dest, content)
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
        max_attempts: int = _http.DEFAULT_MAX_ATTEMPTS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        ua = _resolve_user_agent(user_agent)
        self.rate_limiter = rate_limiter or sec_rate_limiter()
        self._max_attempts = max_attempts
        self._client = httpx.AsyncClient(
            headers=_http.default_headers(user_agent=ua),
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
            wait=_http.make_retry_wait(),
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
        await asyncio.to_thread(_http.atomic_write_bytes, dest, content)
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
    forms = tuple(form_types)
    with _tracer.start_as_current_span("edgar.fetch_filings_for_cik") as span:
        span.set_attribute("edgar.cik", _pad_cik(cik))
        span.set_attribute("edgar.form_types", ",".join(forms))
        owns_client = client is None
        if owns_client:
            client = EdgarClient()
        assert client is not None
        try:
            filings = client.list_recent_filings(cik, form_types=forms)
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
            span.set_attribute("edgar.n_filings", len(results))
            span.set_attribute(
                "edgar.n_fetched", sum(1 for r in results if not r.cache_hit)
            )
            span.set_attribute(
                "edgar.n_cache_hits", sum(1 for r in results if r.cache_hit)
            )
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
