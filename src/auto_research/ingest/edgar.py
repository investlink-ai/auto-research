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
- Transient errors (timeouts, 429s, 5xx) retry with exponential backoff
  + jitter; 429 Retry-After is respected.
- Both a sync `EdgarClient` and an `AsyncEdgarClient` are provided;
  they share the same `TokenBucket` instance when both are used in the
  same process (SEC throttles per-IP).

Idempotency is durable: the manifest is the cache key. Each fetched
filing's row is appended in a `try/finally` block so a mid-loop crash
still records the work that succeeded, preventing wasteful re-fetches
on resume.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import httpx
from tenacity import (
    AsyncRetrying,
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
# Async fan-out: cap simultaneous in-flight requests so we don't queue up a
# thundering herd inside the rate limiter. The bucket would still throttle
# them, but a smaller semaphore keeps memory and connection-pool pressure
# predictable.
_DEFAULT_ASYNC_CONCURRENCY: Final = 4


class EdgarConfigError(RuntimeError):
    """`SEC_USER_AGENT` env var is missing or blank.

    SEC's fair-access policy requires a meaningful UA (name + contact
    email). Without it, requests are throttled or blocked and the
    failure mode is opaque — fail at construction instead.
    """


class EdgarRateLimited(httpx.HTTPStatusError):
    """SEC returned 429. Carries the `Retry-After` hint if present."""

    def __init__(self, response: httpx.Response, retry_after: float | None) -> None:
        super().__init__("SEC rate-limited (429)", request=response.request, response=response)
        self.retry_after = retry_after


class EdgarServerError(httpx.HTTPStatusError):
    """SEC returned a 5xx — transient by assumption, eligible for retry."""


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
)


@dataclass(frozen=True, slots=True)
class Filing:
    cik: str  # zero-padded 10-digit (EDGAR canonical form)
    accession_number: str  # with dashes, e.g. "0001045810-24-000316"
    form_type: str
    primary_document: str
    accepted_datetime: datetime


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


def _classify_response(response: httpx.Response) -> None:
    """Raise a typed retryable error on 429/5xx; leave other status codes alone.

    Called after the HTTP layer settles but BEFORE the body is consumed,
    so that callers' `raise_for_status()` still works on success paths.
    """
    if response.status_code == 429:
        retry_after_raw = response.headers.get("Retry-After")
        retry_after: float | None
        try:
            retry_after = float(retry_after_raw) if retry_after_raw is not None else None
        except ValueError:
            retry_after = None
        raise EdgarRateLimited(response, retry_after)
    if 500 <= response.status_code < 600:
        raise EdgarServerError(
            f"SEC returned {response.status_code}",
            request=response.request,
            response=response,
        )


class EdgarClient:
    """Sync HTTP client with rate limiting + retries.

    A single client instance shares one `TokenBucket` across all
    methods. If you also need an async client in the same process,
    construct it with `rate_limiter=this_sync_client.rate_limiter` so
    both share the SEC fair-access budget.
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
        """One rate-limited GET. Raises retryable errors on 429/5xx."""
        self.rate_limiter.wait()
        resp = self._client.get(url)
        # Honor Retry-After before tenacity's exponential backoff — sleeping
        # here under the lock guarantees the next retry waits at least that
        # long, regardless of the backoff schedule.
        if resp.status_code == 429:
            retry_after_raw = resp.headers.get("Retry-After")
            try:
                if retry_after_raw is not None:
                    import time as _time

                    _time.sleep(float(retry_after_raw))
            except ValueError:
                pass
        _classify_response(resp)
        return resp

    def _get(self, url: str) -> httpx.Response:
        """Rate-limited GET with retries on transient failures."""
        for attempt in Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(
                initial=_DEFAULT_INITIAL_BACKOFF, max=_DEFAULT_MAX_BACKOFF
            ),
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
        """Return all `recent` submissions matching `form_types`.

        Uses only `filings.recent`; older paginated `filings.files[*]`
        entries are out of scope for v1 — `recent` (latest ~1000) is the
        rolling ~5-year window we need.

        Parallel arrays are zipped with `strict=True`, so a malformed
        SEC payload with mismatched array lengths raises a clear
        `ValueError` rather than corrupting field alignment.
        """
        cik_padded = _pad_cik(cik)
        resp = self._get(f"{_SUBMISSIONS_BASE}/submissions/CIK{cik_padded}.json")
        resp.raise_for_status()
        body = resp.json()
        recent = body["filings"]["recent"]
        return _parse_recent(cik_padded, recent, form_types)

    def fetch_filing(self, filing: Filing, *, raw_root: Path) -> tuple[Path, str, bytes]:
        """Download the filing's primary document.

        Atomic on disk: bytes are written to `<dest>.<pid>.tmp` and
        renamed into place after a successful download. A SIGKILL
        mid-write leaves the tempfile, not a truncated canonical file.
        """
        accession_nodash = filing.accession_number.replace("-", "")
        cik_unpadded = str(int(filing.cik))  # Archives URL uses unpadded CIK
        url = (
            f"{_ARCHIVES_BASE}/Archives/edgar/data/{cik_unpadded}"
            f"/{accession_nodash}/{filing.primary_document}"
        )
        resp = self._get(url)
        resp.raise_for_status()
        content = resp.content
        sha = hashlib.sha256(content).hexdigest()
        dest = _destination_path(filing, raw_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".{dest.name}.{os.getpid()}.tmp"
        tmp.write_bytes(content)
        os.replace(tmp, dest)
        return dest, sha, content


class AsyncEdgarClient:
    """Async counterpart to `EdgarClient`. Same rate limit, same retries.

    Useful for fan-out across many CIKs. The rate limiter is shared
    across coroutines via a `threading.Lock` internally, so the 8 r/s
    budget is global to the client, not per-coroutine.
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
        if resp.status_code == 429:
            retry_after_raw = resp.headers.get("Retry-After")
            try:
                if retry_after_raw is not None:
                    await asyncio.sleep(float(retry_after_raw))
            except ValueError:
                pass
        _classify_response(resp)
        return resp

    async def _get(self, url: str) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(
                initial=_DEFAULT_INITIAL_BACKOFF, max=_DEFAULT_MAX_BACKOFF
            ),
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
        accession_nodash = filing.accession_number.replace("-", "")
        cik_unpadded = str(int(filing.cik))
        url = (
            f"{_ARCHIVES_BASE}/Archives/edgar/data/{cik_unpadded}"
            f"/{accession_nodash}/{filing.primary_document}"
        )
        resp = await self._get(url)
        resp.raise_for_status()
        content = resp.content
        sha = hashlib.sha256(content).hexdigest()
        dest = _destination_path(filing, raw_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".{dest.name}.{os.getpid()}.tmp"
        tmp.write_bytes(content)
        os.replace(tmp, dest)
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
    taken once at entry, so the per-filing check is O(1) regardless of
    manifest size. New rows are appended to the manifest in a
    `try/finally` block so a mid-loop crash still persists what
    succeeded — preventing wasteful re-fetches on resume.

    `raw_root` is the project's `data/raw/` root; output is nested
    under `raw_root/edgar/{cik}/{year}/{accession}.{ext}`.
    """
    owns_client = client is None
    if owns_client:
        client = EdgarClient()
    assert client is not None  # narrowed for mypy
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
            # Flush whatever has accumulated even if the loop raised — so a
            # crash mid-fan-out preserves the work that succeeded.
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

    The shared rate limiter still caps total throughput at 8 req/s; the
    semaphore caps simultaneously in-flight downloads so we don't queue
    a thundering herd inside the bucket (which would otherwise wake
    them all in sequence and burst the connection pool).
    """
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
            async with sem:
                assert client is not None  # narrowed for the closure
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

        try:
            # gather raises the first exception but lets the others continue;
            # we still flush whatever completed in `finally`. return_exceptions
            # is False because the orchestrator should surface the failure;
            # the manifest flush in `finally` provides the durability story.
            await asyncio.gather(*(_one(f) for f in deduped))
        finally:
            if new_rows:
                manifest.append(manifest_path, new_rows)

        return [*cached_results, *fetched_results]
    finally:
        if owns_client:
            await client.aclose()


# ---------- helpers ----------


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
    recent: dict[str, list[str]],
    form_types: Iterable[str],
) -> list[Filing]:
    """Zip parallel arrays with `strict=True` so misalignment fails loudly.

    Skips entries with empty `primaryDocument` — those are paper-filed
    submissions where the Archives endpoint would return a directory
    index, which we don't want to silently record as a filing.
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
        if form not in wanted or not primary:
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
    Eastern. We honor that: naive → America/New_York → UTC. Refuses to
    silently misclassify after-hours filings into the wrong trading
    day (INV-1 adjacent).
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
    "EdgarRateLimited",
    "EdgarServerError",
    "FetchResult",
    "Filing",
    "afetch_filings_for_cik",
    "fetch_filings_for_cik",
]
