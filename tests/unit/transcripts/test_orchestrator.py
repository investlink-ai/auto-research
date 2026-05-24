"""Unit tests for the `fetch_transcript` orchestrator.

All external pieces are fakes / mocks:

- `AudioSource` is a hand-rolled stub with controllable `find_audio_url`
  and `download` behavior.
- `WhisperEngine` is a stub that returns a canned `Transcript` without
  calling OpenAI or ffmpeg.

The orchestrator's job: registry lookup → source discovery → atomic
write → Whisper → manifest. These tests verify each branch (happy,
unregistered, source-says-no, empty body, idempotent rerun) without
hitting any external service.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auto_research.ingest import manifest
from auto_research.ingest.transcripts import (
    Transcript,
    fetch_transcript,
    registry,
)


class _FakeSource:
    name = "direct_mp3"

    def __init__(self, audio_url: str | None, payload: bytes = b"FAKE_AUDIO") -> None:
        self._audio_url = audio_url
        self._payload = payload
        self.find_calls: list[tuple[str, int, int]] = []
        self.download_calls: list[str] = []

    def find_audio_url(self, ticker: str, year: int, quarter: int) -> str | None:
        self.find_calls.append((ticker, year, quarter))
        return self._audio_url

    def download(self, audio_url: str) -> bytes:
        self.download_calls.append(audio_url)
        return self._payload

    def close(self) -> None:
        pass


class _FakeEngine:
    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript
        self.calls: list[bytes] = []

    def transcribe(
        self,
        audio: bytes,
        *,
        ticker: str,
        year: int,
        quarter: int,
        event_datetime: datetime,
    ) -> Transcript:
        self.calls.append(audio)
        # Echo the call metadata into a new Transcript so the test can
        # assert the orchestrator passed the right ticker/quarter.
        return self._transcript.model_copy(
            update={
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "event_datetime": event_datetime,
            }
        )

    def close(self) -> None:
        pass


def _canned_transcript() -> Transcript:
    return Transcript(
        ticker="ACME",
        year=2024,
        quarter=2,
        event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
        prepared_remarks="prepared",
        q_and_a="qa",
    )


@pytest.fixture
def registered_acme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(registry.REGISTRY, "ACME", "direct_mp3")


# ---------- happy path ----------


def test_fetch_writes_audio_and_manifest_row(
    registered_acme: None, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url="https://example.com/acme-2024Q2.mp3")
    engine = _FakeEngine(_canned_transcript())

    t = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
    )

    assert isinstance(t, Transcript)
    assert t.ticker == "ACME" and t.year == 2024 and t.quarter == 2

    # File landed on disk under the expected path layout.
    dest = raw_root / "transcripts" / "ACME" / "2024" / "ACME-2024Q2.mp3"
    assert dest.exists()
    assert dest.read_bytes() == b"FAKE_AUDIO"

    # Manifest row recorded.
    table = manifest.read(manifest_path)
    assert table.num_rows == 1
    row = {col: table.column(col)[0].as_py() for col in table.schema.names}
    assert row["source"] == "transcripts"
    assert row["entity_id"] == "ACME"
    assert row["doc_id"] == "ACME-2024Q2"
    assert row["form_type"] == "transcript:direct_mp3"
    assert row["status"] == "ok"
    assert row["content_sha256"] == hashlib.sha256(b"FAKE_AUDIO").hexdigest()

    # Source + engine each called exactly once for the discovery/transcribe step.
    assert source.find_calls == [("ACME", 2024, 2)]
    assert source.download_calls == ["https://example.com/acme-2024Q2.mp3"]
    assert engine.calls == [b"FAKE_AUDIO"]


# ---------- error / no-coverage branches ----------


def test_unregistered_ticker_records_error(tmp_path: Path) -> None:
    """Unregistered tickers write a retryable `status='error'` row —
    NOT a permanent no_coverage. When the registry is later populated
    for this ticker, the next fetch must proceed (not hit cache).
    Also: no audio file written to disk (early return before download)."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"

    t = fetch_transcript(
        "NOT_REGISTERED",
        2024,
        1,
        raw_root=raw_root,
        manifest_path=manifest_path,
        event_datetime=datetime(2024, 1, 15, tzinfo=UTC),
    )
    assert t is None

    table = manifest.read(manifest_path)
    assert table.num_rows == 1
    row = {col: table.column(col)[0].as_py() for col in table.schema.names}
    assert row["status"] == "error"
    assert row["form_type"] == "transcript:unregistered"
    assert row["content_sha256"] is None
    assert row["path"] is None
    assert row["event_datetime"] is None
    # No audio file written — unregistered path short-circuits before download.
    assert not raw_root.exists() or not list(raw_root.rglob("*"))


def test_source_returns_none_records_no_coverage(
    registered_acme: None, tmp_path: Path
) -> None:
    """Registered ticker, but source reports no audio for this
    quarter. This IS permanent (the past doesn't change), so it's
    cached as `no_coverage`, distinct from the retryable `error`."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url=None)
    engine = _FakeEngine(_canned_transcript())

    t = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
    )
    assert t is None
    table = manifest.read(manifest_path)
    assert table.num_rows == 1
    row = {col: table.column(col)[0].as_py() for col in table.schema.names}
    assert row["status"] == "no_coverage"
    assert row["form_type"] == "transcript:direct_mp3"
    # Source was queried but never downloaded.
    assert source.find_calls == [("ACME", 2024, 2)]
    assert source.download_calls == []
    assert engine.calls == []


def test_empty_audio_recorded_as_error(
    registered_acme: None, tmp_path: Path
) -> None:
    """A source that returns empty bytes is recorded as a retryable
    error — NOT a permanent no_coverage. A transient empty body
    shouldn't lock the ticker out forever."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url="https://example.com/a.mp3", payload=b"")
    engine = _FakeEngine(_canned_transcript())

    t = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
    )
    assert t is None
    row = {
        col: manifest.read(manifest_path).column(col)[0].as_py()
        for col in manifest.read(manifest_path).schema.names
    }
    assert row["status"] == "error"
    # Engine was never invoked.
    assert engine.calls == []


def test_error_row_does_not_cache(
    registered_acme: None, tmp_path: Path
) -> None:
    """A previous `status='error'` row must NOT short-circuit the
    next fetch. After the underlying issue (registry, transient
    empty body) clears, the retry should succeed."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    event_dt = datetime(2024, 5, 22, 20, 30, tzinfo=UTC)

    # First run: source returns empty → error row.
    source_empty = _FakeSource(audio_url="https://example.com/a.mp3", payload=b"")
    engine = _FakeEngine(_canned_transcript())
    fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source_empty,
        engine=engine,
        event_datetime=event_dt,
    )
    assert engine.calls == [], "engine must not be invoked on empty body"

    # Second run: source now returns real audio → must NOT be cached.
    source_ok = _FakeSource(audio_url="https://example.com/a.mp3", payload=b"REAL")
    t = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source_ok,
        engine=engine,
        event_datetime=event_dt,
    )
    assert isinstance(t, Transcript), "error row should not cache"
    assert engine.calls == [b"REAL"]


def test_distinct_failure_modes_coexist_in_manifest(
    registered_acme: None, tmp_path: Path
) -> None:
    """`(source, doc_id, status, form_type)` dedup lets a prior
    `transcript:unregistered` error row coexist with a subsequent
    `transcript:direct_mp3` error row — different failure modes
    must not silently overwrite each other."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    event_dt = datetime(2024, 5, 22, 20, 30, tzinfo=UTC)

    # Step 1: pre-seed an "unregistered" error row by fetching a
    # ticker before its registry entry exists.
    fetch_transcript(
        "OTHER",
        2024,
        1,
        raw_root=raw_root,
        manifest_path=manifest_path,
        event_datetime=event_dt,
    )
    # Step 2: same doc_id space but now with empty body from the
    # registered source — different form_type.
    source = _FakeSource(audio_url="https://example.com/a.mp3", payload=b"")
    engine = _FakeEngine(_canned_transcript())
    fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=event_dt,
    )
    # Both error rows must persist.
    table = manifest.read(manifest_path)
    form_types = set(table.column("form_type").to_pylist())
    assert {"transcript:unregistered", "transcript:direct_mp3"} <= form_types


# ---------- idempotency ----------


def test_rerun_is_noop_after_ok_row(registered_acme: None, tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url="https://example.com/a.mp3")
    engine = _FakeEngine(_canned_transcript())
    event_dt = datetime(2024, 5, 22, 20, 30, tzinfo=UTC)

    fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=event_dt,
    )
    assert engine.calls == [b"FAKE_AUDIO"]
    # Second call: cached. Returns None; doesn't touch source/engine.
    t2 = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=event_dt,
    )
    assert t2 is None
    assert source.find_calls == [("ACME", 2024, 2)]  # only the first call
    assert engine.calls == [b"FAKE_AUDIO"]  # only the first call
    # Manifest still has 1 row.
    assert manifest.read(manifest_path).num_rows == 1


def test_rerun_is_noop_after_no_coverage_row(
    registered_acme: None, tmp_path: Path
) -> None:
    """`no_coverage` rows are PERMANENT cache hits — re-trying would
    burn rate budget on a known-empty result."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url=None)
    engine = _FakeEngine(_canned_transcript())
    event_dt = datetime(2024, 5, 22, 20, 30, tzinfo=UTC)

    t1 = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=event_dt,
    )
    assert t1 is None
    t2 = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=event_dt,
    )
    assert t2 is None
    assert source.find_calls == [("ACME", 2024, 2)]  # only the first call
    assert manifest.read(manifest_path).num_rows == 1


# ---------- INV-1 (PIT) discipline ----------


def test_fetch_requires_event_datetime_at_call_site(
    registered_acme: None, tmp_path: Path
) -> None:
    """`event_datetime` is keyword-required at the signature — a
    caller that omits it gets a TypeError at the call site, BEFORE
    any I/O. This is stricter than a runtime fallback-rejection
    (which would still pay bandwidth) and stricter than the
    Transcript-level tz validator (which fires after Whisper)."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url="https://example.com/a.mp3")
    engine = _FakeEngine(_canned_transcript())

    with pytest.raises(TypeError, match="event_datetime"):
        fetch_transcript(  # type: ignore[call-arg]
            "ACME",
            2024,
            2,
            raw_root=raw_root,
            manifest_path=manifest_path,
            source=source,
            engine=engine,
            # event_datetime intentionally omitted
        )
    # No I/O happened — source never even queried.
    assert source.find_calls == []
    assert source.download_calls == []
    assert engine.calls == []


# ---------- _destination_path edge cases ----------


def test_destination_path_strips_url_fragment(
    registered_acme: None, tmp_path: Path
) -> None:
    """A URL like `https://host/audio.mp3#t=120` must NOT leak the
    fragment into the on-disk filename — and the WRONG path (with
    fragment baked in) must NOT exist."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url="https://example.com/acme.mp3#t=120,3600")
    engine = _FakeEngine(_canned_transcript())

    fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
    )
    dest = raw_root / "transcripts" / "ACME" / "2024" / "ACME-2024Q2.mp3"
    wrong = raw_root / "transcripts" / "ACME" / "2024" / "ACME-2024Q2.mp3#t=120,3600"
    assert dest.exists()
    assert not wrong.exists(), "fragment must not leak into filename"


def test_destination_path_strips_path_parameters(
    registered_acme: None, tmp_path: Path
) -> None:
    """RFC 3986 `;jsessionid=…` path parameters (still seen on legacy
    Java IR app servers) must not leak into the on-disk filename."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(
        audio_url="https://example.com/acme.mp3;jsessionid=ABC123?token=xyz"
    )
    engine = _FakeEngine(_canned_transcript())

    fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
        event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
    )
    dest = raw_root / "transcripts" / "ACME" / "2024" / "ACME-2024Q2.mp3"
    assert dest.exists()


# ---------- resource cleanup on failure ----------


class _FakeEngineThatRaises:
    """An engine whose `transcribe` raises — verifies the source is
    still closed via the outer finally."""

    def __init__(self) -> None:
        self.closed = False

    def transcribe(self, *args: object, **kwargs: object) -> Transcript:
        raise RuntimeError("simulated transcription failure")

    def close(self) -> None:
        self.closed = True


def test_orchestrator_does_not_close_caller_injected_resources(
    registered_acme: None, tmp_path: Path
) -> None:
    """Caller-injected `source` and `engine` are NOT owned by the
    orchestrator — even on a transcription failure, the orchestrator
    must leave their close() to the caller. (`owns_source`/`owns_engine`
    are tied to whether the orchestrator constructed them.)"""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"

    closed: dict[str, bool] = {"source": False}

    class _TrackingSource(_FakeSource):
        def close(self) -> None:
            closed["source"] = True

    source = _TrackingSource(audio_url="https://example.com/a.mp3")
    engine = _FakeEngineThatRaises()

    with pytest.raises(RuntimeError, match="simulated"):
        fetch_transcript(
            "ACME",
            2024,
            2,
            raw_root=raw_root,
            manifest_path=manifest_path,
            source=source,
            engine=engine,
            event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
        )
    assert closed["source"] is False, (
        "orchestrator must not close caller-injected resources"
    )


def test_orchestrator_owned_source_closed_when_engine_fails(
    registered_acme: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the orchestrator constructs the source itself, a
    transcription failure must still close it via the outer
    finally (source-leak immunity)."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"

    closes: list[bool] = []

    class _TrackedSource(_FakeSource):
        def close(self) -> None:
            closes.append(True)

    from auto_research.ingest.transcripts import _SOURCE_FACTORIES

    monkeypatch.setitem(
        _SOURCE_FACTORIES,
        "direct_mp3",
        lambda: _TrackedSource(audio_url="https://example.com/a.mp3"),
    )

    with pytest.raises(RuntimeError, match="simulated"):
        fetch_transcript(
            "ACME",
            2024,
            2,
            raw_root=raw_root,
            manifest_path=manifest_path,
            engine=_FakeEngineThatRaises(),
            event_datetime=datetime(2024, 5, 22, 20, 30, tzinfo=UTC),
        )
    assert closes == [True], "orchestrator-owned source was leaked"


def test_transcribe_failure_writes_error_row_for_orphan_audio(
    registered_acme: None, tmp_path: Path
) -> None:
    """When transcription fails AFTER atomic_write_bytes has landed
    audio on disk, the orchestrator must write an `error` row so the
    orphan file is operationally tracked — and a subsequent retry
    must still proceed (error rows don't cache)."""
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url="https://example.com/a.mp3")
    engine = _FakeEngineThatRaises()
    event_dt = datetime(2024, 5, 22, 20, 30, tzinfo=UTC)

    with pytest.raises(RuntimeError, match="simulated"):
        fetch_transcript(
            "ACME",
            2024,
            2,
            raw_root=raw_root,
            manifest_path=manifest_path,
            source=source,
            engine=engine,
            event_datetime=event_dt,
        )

    # Audio file lives on disk (orphan).
    dest = raw_root / "transcripts" / "ACME" / "2024" / "ACME-2024Q2.mp3"
    assert dest.exists()

    # Manifest tracks the failed attempt — operator can find the
    # orphan via the error row's existence.
    table = manifest.read(manifest_path)
    statuses = table.column("status").to_pylist()
    assert "error" in statuses
    row_idx = statuses.index("error")
    assert table.column("form_type")[row_idx].as_py() == "transcript:direct_mp3"

    # A retry with a working engine must succeed (error doesn't cache).
    good_engine = _FakeEngine(_canned_transcript())
    t = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=_FakeSource(audio_url="https://example.com/a.mp3"),
        engine=good_engine,
        event_datetime=event_dt,
    )
    assert isinstance(t, Transcript)
