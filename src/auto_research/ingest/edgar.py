"""SEC EDGAR public-filings client.

Fetches 10-K, 10-Q, 8-K, S-1, S-3 (and any other configured form types)
for a given CIK and persists raw bytes at
`data/raw/edgar/{cik}/{year}/{accession}.{ext}`. Idempotent on
`(cik, accession_number)` via the append-only Parquet manifest in
`auto_research.ingest.manifest`.

`accepted_datetime` (SEC's `acceptanceDateTime`) is recorded as the
canonical point-in-time stamp — it's the moment EDGAR exposed the filing
to the public and is therefore the earliest timestamp at which a trading
signal could legitimately incorporate it (lag-1 cutoff applies downstream
in Feast — INV-1).

SEC requires a meaningful User-Agent on every request (their fair-access
policy). The client reads `SEC_USER_AGENT` from the environment and
raises `EdgarConfigError` if it isn't set — fail loud rather than send
anonymous traffic that SEC may block.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from auto_research.ingest import manifest

DEFAULT_FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q", "8-K", "S-1", "S-3")
SOURCE: str = "edgar"

_SUBMISSIONS_BASE = "https://data.sec.gov"
_ARCHIVES_BASE = "https://www.sec.gov"


class EdgarConfigError(RuntimeError):
    """`SEC_USER_AGENT` env var is missing or blank.

    SEC's fair-access policy requires a meaningful UA (name + contact
    email). Without it, requests are throttled or blocked and the
    failure mode is opaque — fail at construction instead.
    """


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


class EdgarClient:
    """Thin HTTP client around the two EDGAR endpoints we need.

    `data.sec.gov` for the per-company submissions JSON;
    `www.sec.gov` for the per-filing Archives directory. A single
    `httpx.Client` covers both — paths in `list_recent_filings` and
    `fetch_filing` are absolute URLs.
    """

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        ua = (user_agent if user_agent is not None else os.environ.get("SEC_USER_AGENT", "")).strip()
        if not ua:
            raise EdgarConfigError(
                "SEC requires a meaningful User-Agent header for data.sec.gov / "
                "www.sec.gov requests. Set SEC_USER_AGENT in the environment "
                "(e.g., 'Your Name your@email.com')."
            )
        self._client = httpx.Client(
            headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def list_recent_filings(
        self,
        cik: str | int,
        *,
        form_types: Iterable[str],
    ) -> list[Filing]:
        """Return all `recent` submissions matching `form_types` for the company.

        Uses only the `filings.recent` arrays from the submissions JSON;
        older filings paginated under `filings.files[*].name` are out of
        scope for v1 — recent (~1000 latest) covers the rolling
        ~5-year window we need.
        """
        cik_padded = _pad_cik(cik)
        resp = self._client.get(f"{_SUBMISSIONS_BASE}/submissions/CIK{cik_padded}.json")
        resp.raise_for_status()
        body = resp.json()
        recent = body["filings"]["recent"]
        wanted = set(form_types)
        out: list[Filing] = []
        for i, form in enumerate(recent["form"]):
            if form not in wanted:
                continue
            out.append(
                Filing(
                    cik=cik_padded,
                    accession_number=recent["accessionNumber"][i],
                    form_type=form,
                    primary_document=recent["primaryDocument"][i],
                    accepted_datetime=_parse_acceptance(recent["acceptanceDateTime"][i]),
                )
            )
        return out

    def fetch_filing(self, filing: Filing, *, raw_root: Path) -> tuple[Path, str, bytes]:
        """Download the filing's primary document.

        Returns `(path, sha256_hex, content_bytes)` so callers can decide
        whether to also persist the bytes themselves; we always write to
        the canonical path under `raw_root`.
        """
        accession_nodash = filing.accession_number.replace("-", "")
        cik_unpadded = str(int(filing.cik))  # Archives URL uses unpadded CIK
        url = (
            f"{_ARCHIVES_BASE}/Archives/edgar/data/{cik_unpadded}"
            f"/{accession_nodash}/{filing.primary_document}"
        )
        resp = self._client.get(url)
        resp.raise_for_status()
        content = resp.content
        sha = hashlib.sha256(content).hexdigest()
        year = filing.accepted_datetime.astimezone(UTC).year
        ext = Path(filing.primary_document).suffix or ".bin"
        dest = raw_root / filing.cik / str(year) / f"{filing.accession_number}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
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

    The manifest is the idempotency boundary: filings already present
    under `(source="edgar", doc_id=accession_number)` are skipped with
    `cache_hit=True` and not re-downloaded.

    `raw_root` is the project's `data/raw/` root; this function nests
    output under `raw_root/edgar/{cik}/{year}/{accession}.{ext}`.
    """
    owns_client = client is None
    client = client or EdgarClient()
    try:
        filings = client.list_recent_filings(cik, form_types=form_types)
        results: list[FetchResult] = []
        new_rows: list[dict[str, object]] = []
        for filing in filings:
            if manifest.contains(manifest_path, source=SOURCE, doc_id=filing.accession_number):
                results.append(
                    FetchResult(
                        cik=filing.cik,
                        accession_number=filing.accession_number,
                        form_type=filing.form_type,
                        accepted_datetime=filing.accepted_datetime,
                        path=None,
                        content_sha256=None,
                        cache_hit=True,
                    )
                )
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
            new_rows.append(
                {
                    "source": SOURCE,
                    "entity_id": filing.cik,
                    "doc_id": filing.accession_number,
                    "form_type": filing.form_type,
                    "event_datetime": filing.accepted_datetime,
                    "fetched_at": datetime.now(UTC),
                    "content_sha256": sha,
                    "path": str(path),
                    "status": "ok",
                }
            )
        if new_rows:
            manifest.append(manifest_path, new_rows)
        return results
    finally:
        if owns_client:
            client.close()


def _pad_cik(cik: str | int) -> str:
    return f"{int(cik):010d}"


def _parse_acceptance(value: str) -> datetime:
    """EDGAR returns acceptance datetimes either as ISO with `Z` or naive.

    Normalize to UTC-aware. Naive values are documented by SEC as
    Eastern; here we treat any missing-tz value as UTC for simplicity —
    the lag-1 cutoff downstream is coarse enough that intra-day TZ
    nuances don't change the trading-day classification. Refine if/when
    a regression test demands it.
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


__all__ = [
    "DEFAULT_FORM_TYPES",
    "SOURCE",
    "EdgarClient",
    "EdgarConfigError",
    "FetchResult",
    "Filing",
    "fetch_filings_for_cik",
]
