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


# ---------- no-coverage branches ----------


def test_unregistered_ticker_records_no_coverage(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"

    t = fetch_transcript(
        "NOT_REGISTERED",
        2024,
        1,
        raw_root=raw_root,
        manifest_path=manifest_path,
    )
    assert t is None

    table = manifest.read(manifest_path)
    assert table.num_rows == 1
    row = {col: table.column(col)[0].as_py() for col in table.schema.names}
    assert row["status"] == "no_coverage"
    assert row["form_type"] == "transcript:unregistered"
    assert row["content_sha256"] is None
    assert row["path"] is None
    assert row["event_datetime"] is None


def test_source_returns_none_records_no_coverage(
    registered_acme: None, tmp_path: Path
) -> None:
    """Registered ticker, but source reports no audio for this quarter."""
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


def test_empty_audio_recorded_as_no_coverage(
    registered_acme: None, tmp_path: Path
) -> None:
    """A source that returns empty bytes (slipped past its own
    retries) is bucketed as no_coverage rather than a sha-of-empty
    cache hit."""
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
    )
    assert t is None
    row = {
        col: manifest.read(manifest_path).column(col)[0].as_py()
        for col in manifest.read(manifest_path).schema.names
    }
    assert row["status"] == "no_coverage"
    # Engine was never invoked.
    assert engine.calls == []


# ---------- idempotency ----------


def test_rerun_is_noop_after_ok_row(registered_acme: None, tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.parquet"
    source = _FakeSource(audio_url="https://example.com/a.mp3")
    engine = _FakeEngine(_canned_transcript())

    fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
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

    t1 = fetch_transcript(
        "ACME",
        2024,
        2,
        raw_root=raw_root,
        manifest_path=manifest_path,
        source=source,
        engine=engine,
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
    )
    assert t2 is None
    assert source.find_calls == [("ACME", 2024, 2)]  # only the first call
    assert manifest.read(manifest_path).num_rows == 1
