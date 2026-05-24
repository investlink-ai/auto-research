# Issue #52 — feat(observability): cross-cutting OTel spans

> **For agentic workers:** Execute task-by-task with TDD discipline per
> `docs/AI_WORKFLOW.md`. Steps use checkbox (`- [ ]`) syntax for tracking.
> Repo authority (`AGENTS.md`) wins where it conflicts with skill defaults.

**Goal:** Close the gap between the documented observability strategy
(`docs/specs/...:§15`) and the running system: every `auto-research` CLI
process initializes telemetry; every orchestration boundary in ingest +
the first extraction worker emits a manual span; tests prove spans get
created without needing a live Langfuse.

**Architecture:** Add `try_init_telemetry()` as an env-tolerant wrapper
so CLI entry points stay usable when Langfuse isn't running locally.
Wrap four orchestration entry points in manual spans (issue #52) plus
one extract-worker span so `extract/client.py:151`'s existing
`llm.cost.est_usd` attribute lands under a named parent. Test via
in-memory OTel exporter — no live OTLP needed in CI; the existing
`tests/integration/test_telemetry_export.py` covers delivery to Langfuse.

**Tech stack:** `opentelemetry-api` + `opentelemetry-sdk` (already
transitively present via `traceloop-sdk`), pytest fixtures, Click
CliRunner, the established VCR / fake / MockTransport patterns.

---

## File map

**Source — production code touched:**

- `src/auto_research/telemetry.py` — add `try_init_telemetry()` helper.
- `src/auto_research/cli.py` — call `try_init_telemetry()` in
  `ingest_edgar` and `extract_s_filings` command bodies.
- `src/auto_research/ingest/edgar.py` — wrap `fetch_filings_for_cik`
  in `edgar.fetch_filings_for_cik` span.
- `src/auto_research/ingest/transcripts/__init__.py` — wrap
  `fetch_transcript` in `transcript.fetch` span.
- `src/auto_research/ingest/transcripts/sources/youtube.py` — wrap
  `find_audio_url` (→ `transcript.find_audio_url`) and `download`
  (→ `transcript.download`).
- `src/auto_research/ingest/transcripts/sources/direct_mp3.py` —
  wrap `download` (→ `transcript.download`).
- `src/auto_research/extract/workers/s_filings.py` — wrap
  `extract_s_filing` in `extract.s_filings` span (parents the
  existing `llm.cost.est_usd` emission from `extract/client.py:151`).

**Tests — new + extended:**

- `tests/conftest.py` (new) — `span_recorder` fixture (per-test
  `InMemorySpanExporter` + `TracerProvider`).
- `tests/unit/test_telemetry.py` — extend with `try_init_telemetry`
  cases.
- `tests/unit/test_cli.py` — assert CLI commands call init helper.
- `tests/unit/test_edgar.py` — assert span emission for
  `fetch_filings_for_cik`.
- `tests/unit/transcripts/test_orchestrator.py` — assert
  `transcript.fetch` span across cache / no_coverage / error / ok
  outcomes.
- `tests/unit/transcripts/test_youtube.py` — assert
  `transcript.find_audio_url` + `transcript.download` spans.
- `tests/unit/transcripts/test_direct_mp3.py` — assert
  `transcript.download` span.
- `tests/unit/test_extract_worker_s_filings.py` — assert
  `extract.s_filings` parent span.

**Docs:**

- `docs/ARCHITECTURE.md` — add §N. Observability subsection.

**Issue:**

- GitHub `#52` — edit body to incorporate the polish notes.

---

## Span / attribute conventions

| Span name | Emitter | Attributes |
|---|---|---|
| `edgar.fetch_filings_for_cik` | `ingest/edgar.py` | `edgar.cik`, `edgar.form_types` (comma-joined), `edgar.n_filings`, `edgar.n_fetched`, `edgar.n_cache_hits` |
| `transcript.fetch` | `ingest/transcripts/__init__.py` | `transcript.ticker`, `transcript.year`, `transcript.quarter`, `transcript.source_name` (string; `"unregistered"` when no registry entry), `transcript.outcome` (`cached` / `unregistered` / `no_coverage` / `ok` / `error`) |
| `transcript.find_audio_url` | `youtube.py` | `transcript.ticker`, `transcript.year`, `transcript.quarter`, `transcript.query`, `transcript.result_count`, `transcript.matched` (bool) |
| `transcript.download` | `youtube.py` + `direct_mp3.py` | `transcript.source_name`, `transcript.bytes`, `transcript.duration_ms` |
| `extract.s_filings` | `extract/workers/s_filings.py` | `extract.worker` (`s_filings`), `extract.doc_id`, `extract.outcome` (`cache_hit` / `persisted` / `quarantined`) |

Outcomes are recorded by `span.set_attribute("…outcome", …)`. Span
status is set with `span.set_status(Status(StatusCode.ERROR, …))` only
when an exception propagates from the wrapped block (otherwise the
default UNSET, which the OTel UI renders as OK).

---

## Task 1 — `try_init_telemetry()` helper + tests

**Files:**
- Modify: `src/auto_research/telemetry.py:1` (extend module)
- Modify: `tests/unit/test_telemetry.py:1` (extend)

- [ ] **Step 1.1: Write failing tests for `try_init_telemetry`**

Append to `tests/unit/test_telemetry.py`:

```python
def test_try_init_telemetry_returns_false_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing env -> warn once to stderr, return False, don't raise."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from auto_research import telemetry as t

    monkeypatch.setattr(t, "_INITIALIZED", False)
    monkeypatch.setattr(t, "_TRY_INIT_WARNED", False)

    assert t.try_init_telemetry() is False
    captured = capsys.readouterr()
    assert "telemetry" in captured.err.lower()
    # Idempotent warning — calling twice does not double-print.
    assert t.try_init_telemetry() is False
    captured2 = capsys.readouterr()
    assert captured2.err == ""


def test_try_init_telemetry_returns_true_when_already_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_research import telemetry as t

    monkeypatch.setattr(t, "_INITIALIZED", True)
    assert t.try_init_telemetry() is True
```

- [ ] **Step 1.2: Run tests, confirm failure**

```bash
uv run --active pytest tests/unit/test_telemetry.py -v
```

Expected: 2 new tests FAIL with `AttributeError: module … has no attribute 'try_init_telemetry'`.

- [ ] **Step 1.3: Implement `try_init_telemetry()`**

Add to `src/auto_research/telemetry.py` (after the existing
`is_initialized` definition):

```python
_TRY_INIT_WARNED: bool = False


def try_init_telemetry(*, service_name: str = "auto-research") -> bool:
    """Best-effort `init_telemetry()` for CLI / interactive entry points.

    Returns True iff telemetry is initialized (either now or already).
    On `TelemetryNotConfiguredError`, prints a single one-line warning
    to stderr and returns False — the CLI must remain usable without
    a running Langfuse, but operators should know spans aren't being
    emitted. The warning is deduplicated per process via
    `_TRY_INIT_WARNED`; re-running a command does not spam stderr.

    Use this from process-start entry points. Tests and the
    integration smoke continue to call `init_telemetry()` directly
    so a misconfigured environment fails loud, not silently.
    """
    global _TRY_INIT_WARNED
    try:
        init_telemetry(service_name=service_name)
    except TelemetryNotConfiguredError as exc:
        if not _TRY_INIT_WARNED:
            print(
                f"warn: telemetry disabled — {exc}",
                file=sys.stderr,
            )
            _TRY_INIT_WARNED = True
        return False
    return True
```

Add `import sys` to the imports block at the top of the file (next to
`import os`).

- [ ] **Step 1.4: Run tests, confirm pass**

```bash
uv run --active pytest tests/unit/test_telemetry.py -v
```

Expected: all telemetry unit tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/auto_research/telemetry.py tests/unit/test_telemetry.py
git commit -m "feat(telemetry): add try_init_telemetry best-effort wrapper

Refs #52. Strict init_telemetry stays for tests/integration; CLI
entry points get an env-tolerant variant that warns once and
returns bool."
```

---

## Task 2 — `span_recorder` test fixture

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 2.1: Write the fixture**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures for the auto-research test suite.

`span_recorder` installs an in-memory OTel tracer provider for the
duration of a test and exposes the recorded spans. Used by unit tests
that assert on manual instrumentation without needing live OTLP.

`init_telemetry()` is NOT called inside the fixture — the fixture
provides its own provider, and Traceloop's provider would race with
this one if both were installed. Tests that need real export use
`tests/integration/test_telemetry_export.py` instead.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class SpanRecorder:
    """Convenience wrapper around an `InMemorySpanExporter`.

    Tests typically use `recorder.finished_spans()` to get the list of
    completed spans and `recorder.by_name(name)` to filter to one span
    by name.
    """

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        self._exporter = exporter

    def finished_spans(self) -> tuple[ReadableSpan, ...]:
        return tuple(self._exporter.get_finished_spans())

    def by_name(self, name: str) -> tuple[ReadableSpan, ...]:
        return tuple(s for s in self.finished_spans() if s.name == name)

    def one(self, name: str) -> ReadableSpan:
        matches = self.by_name(name)
        assert len(matches) == 1, (
            f"expected exactly one span named {name!r}, "
            f"got {len(matches)}: {[s.name for s in self.finished_spans()]}"
        )
        return matches[0]


@pytest.fixture
def span_recorder() -> Iterator[SpanRecorder]:
    """Install an in-memory tracer provider; yield a SpanRecorder.

    Restores the previous global tracer provider on teardown so tests
    that don't request the fixture observe the OTel default (a no-op
    provider — `extract/client.py:151`'s `get_current_span()` already
    relies on this fallback).
    """
    previous = trace.get_tracer_provider()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # `override=True` is required because OpenTelemetry refuses to
    # replace an already-set provider otherwise. Without it, a prior
    # test that ran Traceloop.init() would lock subsequent tests out.
    trace.set_tracer_provider(provider)
    try:
        yield SpanRecorder(exporter)
    finally:
        provider.shutdown()
        # Restore previous provider — OTel's API doesn't expose an
        # "unset" call; resetting to the proxy default is the
        # closest we can get without monkeypatching internals.
        trace._TRACER_PROVIDER = previous  # type: ignore[attr-defined]
```

- [ ] **Step 2.2: Smoke-test the fixture works**

Add to `tests/unit/test_telemetry.py`:

```python
def test_span_recorder_fixture_captures_spans(span_recorder) -> None:  # type: ignore[no-untyped-def]
    from opentelemetry import trace

    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("smoke") as span:
        span.set_attribute("k", "v")

    spans = span_recorder.finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "smoke"
    assert spans[0].attributes["k"] == "v"
```

- [ ] **Step 2.3: Run, confirm pass**

```bash
uv run --active pytest tests/unit/test_telemetry.py::test_span_recorder_fixture_captures_spans -v
```

Expected: PASS.

- [ ] **Step 2.4: Commit**

```bash
git add tests/conftest.py tests/unit/test_telemetry.py
git commit -m "test(otel): add span_recorder fixture

Refs #52. Per-test InMemorySpanExporter + TracerProvider; restores
the previous provider on teardown. Used by manual-span assertions
across ingest + extract."
```

---

## Task 3 — Wire `try_init_telemetry()` into CLI commands

**Files:**
- Modify: `src/auto_research/cli.py:1` (import) and the bodies of
  `ingest_edgar` (around line 145) and `extract_s_filings` (around
  line 195).
- Modify: `tests/unit/test_cli.py:1`

- [ ] **Step 3.1: Write failing tests**

Add to `tests/unit/test_cli.py` (use the existing `runner` fixture):

```python
def test_ingest_edgar_initializes_telemetry(
    runner: CliRunner, tmp_path: Path
) -> None:
    with (
        patch("auto_research.cli.try_init_telemetry", autospec=True) as mock_init,
        patch("auto_research.cli.fetch_filings_for_cik", autospec=True) as mock_fetch,
    ):
        mock_init.return_value = True
        mock_fetch.return_value = []
        result = runner.invoke(
            cli,
            [
                "ingest",
                "edgar",
                "--cik",
                "0001045810",
                "--raw-root",
                str(tmp_path / "raw"),
                "--manifest-path",
                str(tmp_path / "manifest.parquet"),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_init.assert_called_once()


def test_extract_s_filings_initializes_telemetry(
    runner: CliRunner, tmp_path: Path
) -> None:
    # Empty manifest — extract loop exits immediately, but init still fires.
    manifest = tmp_path / "manifest.parquet"
    empty = pa.table(
        {
            "source": pa.array([], type=pa.string()),
            "entity_id": pa.array([], type=pa.string()),
            "doc_id": pa.array([], type=pa.string()),
            "form_type": pa.array([], type=pa.string()),
            "event_datetime": pa.array([], type=pa.timestamp("us", tz="UTC")),
            "fetched_at": pa.array([], type=pa.timestamp("us", tz="UTC")),
            "content_sha256": pa.array([], type=pa.string()),
            "path": pa.array([], type=pa.string()),
            "status": pa.array([], type=pa.string()),
        }
    )
    pq.write_table(empty, manifest)

    with patch("auto_research.cli.try_init_telemetry", autospec=True) as mock_init:
        mock_init.return_value = False  # missing env path — still callable
        result = runner.invoke(
            cli,
            [
                "extract",
                "s-filings",
                "--cik",
                "0001045810",
                "--manifest-path",
                str(manifest),
                "--out-root",
                str(tmp_path / "extracted"),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_init.assert_called_once()
```

- [ ] **Step 3.2: Run, confirm failure**

```bash
uv run --active pytest tests/unit/test_cli.py -v -k telemetry
```

Expected: FAIL with `AttributeError` or "no attribute try_init_telemetry".

- [ ] **Step 3.3: Implement**

In `src/auto_research/cli.py`:

1. Add import (next to other `auto_research` imports):

```python
from auto_research.telemetry import try_init_telemetry
```

2. At the top of `ingest_edgar` (immediately after `cik =
_normalize_cik(cik)`), add:

```python
    try_init_telemetry()
```

3. At the top of `extract_s_filings` (immediately after `cik =
_normalize_cik(cik)`), add:

```python
    try_init_telemetry()
```

- [ ] **Step 3.4: Run, confirm pass**

```bash
uv run --active pytest tests/unit/test_cli.py -v
```

Expected: all CLI tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/auto_research/cli.py tests/unit/test_cli.py
git commit -m "feat(cli): initialize telemetry in ingest + extract commands

Refs #52. The strategy ratified in spec §15 requires every entry
point to call init_telemetry(); the CLI commands are the W1 entry
points. try_init_telemetry keeps the commands usable without
Langfuse running locally."
```

---

## Task 4 — Instrument `edgar.fetch_filings_for_cik`

**Files:**
- Modify: `src/auto_research/ingest/edgar.py:52` (add tracer
  import + module-level tracer) and `:358` (wrap `fetch_filings_for_cik`).
- Modify: `tests/unit/test_edgar.py:1`.

- [ ] **Step 4.1: Write failing test**

Append to `tests/unit/test_edgar.py`:

```python
def test_fetch_filings_for_cik_emits_span(
    span_recorder, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    """The orchestrator emits one edgar.fetch_filings_for_cik span
    with attribute discipline matching the documented contract."""
    # Reuse the existing FakeEdgarClient / fixture pattern from this
    # module. Substitute the real fixture name when copying — see
    # the surrounding tests for the canonical setup.
    from auto_research.ingest import edgar

    fake_client = _build_fake_client_with_two_filings()  # existing helper
    edgar.fetch_filings_for_cik(
        "0001045810",
        form_types=("S-3",),
        raw_root=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.parquet",
        client=fake_client,
    )

    span = span_recorder.one("edgar.fetch_filings_for_cik")
    assert span.attributes["edgar.cik"] == "0001045810"
    assert span.attributes["edgar.form_types"] == "S-3"
    assert span.attributes["edgar.n_filings"] == 2
    assert span.attributes["edgar.n_fetched"] == 2
    assert span.attributes["edgar.n_cache_hits"] == 0
```

If `_build_fake_client_with_two_filings` doesn't yet exist in the
test module, adapt the pattern used by the nearest existing
`fetch_filings_for_cik` test — the goal is to drive the function
without hitting SEC.

- [ ] **Step 4.2: Run, confirm failure**

```bash
uv run --active pytest tests/unit/test_edgar.py::test_fetch_filings_for_cik_emits_span -v
```

Expected: FAIL — no span recorded.

- [ ] **Step 4.3: Implement**

In `src/auto_research/ingest/edgar.py`:

1. Add tracer (at module top, near `_logger`):

```python
from opentelemetry import trace

_tracer = trace.get_tracer(__name__)
```

2. Wrap the body of `fetch_filings_for_cik` (the entire `try`
block) in a span. The cleanest shape is to push the whole existing
body into a `with` block:

```python
def fetch_filings_for_cik(
    cik: str | int,
    *,
    form_types: Iterable[str] = DEFAULT_FORM_TYPES,
    raw_root: Path,
    manifest_path: Path,
    client: EdgarClient | None = None,
) -> list[FetchResult]:
    """... existing docstring ..."""
    padded = _pad_cik(cik)
    forms = tuple(form_types)
    with _tracer.start_as_current_span("edgar.fetch_filings_for_cik") as span:
        span.set_attribute("edgar.cik", padded)
        span.set_attribute("edgar.form_types", ",".join(forms))
        owns_client = client is None
        if owns_client:
            client = EdgarClient()
        assert client is not None
        try:
            filings = client.list_recent_filings(padded, form_types=forms)
            # ... rest of existing body unchanged, but use `padded` /
            # `forms` and the loop already tracks fetched/cache_hit
            # counts via results.
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
                    path, sha, _ = client.fetch_filing(
                        filing, raw_root=raw_root / SOURCE
                    )
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
                        _manifest_row(filing, path, sha, status="ok")
                    )
            finally:
                if new_rows:
                    manifest.append(manifest_path, new_rows)
            span.set_attribute("edgar.n_filings", len(results))
            span.set_attribute(
                "edgar.n_cache_hits",
                sum(1 for r in results if r.cache_hit),
            )
            span.set_attribute(
                "edgar.n_fetched",
                sum(1 for r in results if not r.cache_hit),
            )
            return results
        finally:
            if owns_client:
                client.close()
```

Notes:
- `_pad_cik(cik)` was previously implicit in `client.list_recent_filings` —
  if the existing body passes `cik` raw, do the same and read `padded`
  off the first filing's `cik` field for the attribute. Match the
  established behavior; do not refactor the padding semantics here.
- Async variant `afetch_filings_for_cik` is out of scope for this
  issue — explicitly noted in §52. Leave it as-is.

- [ ] **Step 4.4: Run, confirm pass**

```bash
uv run --active pytest tests/unit/test_edgar.py -v
```

Expected: all edgar tests PASS (existing + new).

- [ ] **Step 4.5: Commit**

```bash
git add src/auto_research/ingest/edgar.py tests/unit/test_edgar.py
git commit -m "feat(ingest/edgar): emit fetch_filings_for_cik OTel span

Refs #52. Attributes: cik / form_types / n_filings / n_fetched /
n_cache_hits. Async variant out of scope per the issue body."
```

---

## Task 5 — Instrument transcripts `fetch_transcript`

**Files:**
- Modify: `src/auto_research/ingest/transcripts/__init__.py:36` (import +
  module tracer) and `:220` (`fetch_transcript` body).
- Modify: `tests/unit/transcripts/test_orchestrator.py:1`.

- [ ] **Step 5.1: Write failing tests** (cover the five outcomes)

Append to `tests/unit/transcripts/test_orchestrator.py`:

```python
def test_fetch_transcript_emits_span_on_cache_hit(
    span_recorder, _empty_manifest_with_ok_row  # type: ignore[no-untyped-def]
) -> None:
    """Manifest already has an `ok` row → outcome=cached, no source call."""
    # Reuse the existing cache-hit fixture pattern in this module.
    fetch_transcript(...)  # populate args per existing pattern
    span = span_recorder.one("transcript.fetch")
    assert span.attributes["transcript.outcome"] == "cached"


def test_fetch_transcript_outcome_unregistered(
    span_recorder, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    """No registry entry → outcome=unregistered."""
    fetch_transcript(
        "XYZ_UNKNOWN_TICKER",
        2024,
        1,
        raw_root=tmp_path,
        manifest_path=tmp_path / "manifest.parquet",
        event_datetime=datetime(2024, 4, 1, tzinfo=UTC),
    )
    span = span_recorder.one("transcript.fetch")
    assert span.attributes["transcript.outcome"] == "unregistered"
    assert span.attributes["transcript.source_name"] == "unregistered"


def test_fetch_transcript_outcome_no_coverage(
    span_recorder, _registered_ticker, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    """Source returns None → outcome=no_coverage."""
    fake_source = _build_fake_source(find_audio_url_returns=None)
    fetch_transcript(
        _registered_ticker,
        2024,
        1,
        raw_root=tmp_path,
        manifest_path=tmp_path / "manifest.parquet",
        event_datetime=datetime(2024, 4, 1, tzinfo=UTC),
        source=fake_source,
    )
    span = span_recorder.one("transcript.fetch")
    assert span.attributes["transcript.outcome"] == "no_coverage"


def test_fetch_transcript_outcome_error_when_source_raises(
    span_recorder, _registered_ticker, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    fake_source = _build_fake_source(find_audio_url_raises=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        fetch_transcript(
            _registered_ticker,
            2024,
            1,
            raw_root=tmp_path,
            manifest_path=tmp_path / "manifest.parquet",
            event_datetime=datetime(2024, 4, 1, tzinfo=UTC),
            source=fake_source,
        )
    span = span_recorder.one("transcript.fetch")
    assert span.attributes["transcript.outcome"] == "error"
    # ERROR status, not UNSET.
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.ERROR
```

If the helper names above don't exist in `test_orchestrator.py`,
substitute the equivalents already in use; the goal is one assertion
per outcome.

- [ ] **Step 5.2: Run, confirm failure**

```bash
uv run --active pytest tests/unit/transcripts/test_orchestrator.py -v
```

Expected: 4 new tests FAIL.

- [ ] **Step 5.3: Implement**

In `src/auto_research/ingest/transcripts/__init__.py`:

1. Add at the top (near `_logger`):

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_tracer = trace.get_tracer(__name__)
```

2. Wrap the body of `fetch_transcript` in a span. Mark the outcome
on each return / exception path:

```python
def fetch_transcript(
    ticker: str,
    year: int,
    quarter: int,
    *,
    raw_root: Path,
    manifest_path: Path,
    event_datetime: datetime,
    engine: Transcriber | None = None,
    source: AudioSource | None = None,
) -> Transcript | None:
    """... existing docstring ..."""
    ticker_upper = ticker.upper()
    doc_id = _doc_id(ticker_upper, year, quarter)
    with _tracer.start_as_current_span("transcript.fetch") as span:
        span.set_attribute("transcript.ticker", ticker_upper)
        span.set_attribute("transcript.year", year)
        span.set_attribute("transcript.quarter", quarter)

        if manifest.contains(
            manifest_path, source=SOURCE, doc_id=doc_id, status=_CACHED_STATUSES
        ):
            span.set_attribute("transcript.outcome", "cached")
            _logger.debug("transcript cached: %s", doc_id)
            return None

        source_name = registry.lookup(ticker_upper)
        if source_name is None:
            span.set_attribute("transcript.source_name", _UNREGISTERED_SUFFIX)
            span.set_attribute("transcript.outcome", "unregistered")
            _logger.info("no transcript source registered for %s", ticker_upper)
            manifest.append(
                manifest_path,
                [_error_row(ticker_upper, year, quarter, source_name=_UNREGISTERED_SUFFIX)],
                unique_keys=_ERROR_DEDUP_KEYS,
            )
            return None

        span.set_attribute("transcript.source_name", source_name)

        owns_source = source is None
        if source is None:
            source = _open_source(source_name)
        try:
            try:
                audio_url = source.find_audio_url(ticker_upper, year, quarter)
            except Exception as exc:
                span.set_attribute("transcript.outcome", "error")
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                _logger.warning(
                    "source %s raised during find_audio_url for %s %sQ%s — recording error",
                    source_name,
                    ticker_upper,
                    year,
                    quarter,
                )
                manifest.append(
                    manifest_path,
                    [_error_row(ticker_upper, year, quarter, source_name=source_name)],
                    unique_keys=_ERROR_DEDUP_KEYS,
                )
                raise

            if audio_url is None:
                span.set_attribute("transcript.outcome", "no_coverage")
                _logger.info(
                    "source %s reports no coverage for %s %sQ%s",
                    source_name,
                    ticker_upper,
                    year,
                    quarter,
                )
                manifest.append(
                    manifest_path,
                    [_no_coverage_row(ticker_upper, year, quarter, source_name=source_name)],
                )
                return None

            # ... rest of existing body unchanged until success ...
            # At the existing success-return site, before returning:
            #   span.set_attribute("transcript.outcome", "ok")
            #   return transcript
            #
            # If a later branch raises (engine failures, download
            # errors), wrap them with:
            #   span.set_attribute("transcript.outcome", "error")
            #   span.set_status(Status(StatusCode.ERROR, str(exc)))
            #   raise
        finally:
            if owns_source:
                source.close()
```

Apply the same outcome-setting / status pattern to the
download-and-transcribe section of the existing body. Reference
file: `src/auto_research/ingest/transcripts/__init__.py:220-380` (the
existing function body; precise line numbers depend on the as-merged
state at pickup).

- [ ] **Step 5.4: Run, confirm pass**

```bash
uv run --active pytest tests/unit/transcripts/test_orchestrator.py -v
```

Expected: all PASS.

- [ ] **Step 5.5: Commit**

```bash
git add src/auto_research/ingest/transcripts/__init__.py tests/unit/transcripts/test_orchestrator.py
git commit -m "feat(transcripts): emit transcript.fetch OTel span

Refs #52. Five outcomes covered: cached / unregistered /
no_coverage / ok / error. Error status is set when a source raises
during find_audio_url or download."
```

---

## Task 6 — Instrument `YouTubeSource.find_audio_url` + `.download`

**Files:**
- Modify: `src/auto_research/ingest/transcripts/sources/youtube.py:40`
  (import + tracer) and the two method bodies.
- Modify: `tests/unit/transcripts/test_youtube.py:1`.

- [ ] **Step 6.1: Write failing tests**

Append to `tests/unit/transcripts/test_youtube.py`:

```python
def test_find_audio_url_emits_span_with_matched_true(
    span_recorder,  # type: ignore[no-untyped-def]
) -> None:
    """A matched search result records matched=True + result_count."""
    fake_factory = _build_fake_factory_returning_matching_entry()
    source = YouTubeSource(factory=fake_factory, verify_yt_dlp=False)
    url = source.find_audio_url("NVDA", 2024, 1)
    assert url is not None

    span = span_recorder.one("transcript.find_audio_url")
    assert span.attributes["transcript.ticker"] == "NVDA"
    assert span.attributes["transcript.year"] == 2024
    assert span.attributes["transcript.quarter"] == 1
    assert span.attributes["transcript.matched"] is True
    assert span.attributes["transcript.result_count"] >= 1


def test_find_audio_url_emits_span_matched_false(
    span_recorder,  # type: ignore[no-untyped-def]
) -> None:
    fake_factory = _build_fake_factory_returning_no_match()
    source = YouTubeSource(factory=fake_factory, verify_yt_dlp=False)
    url = source.find_audio_url("NVDA", 2024, 1)
    assert url is None
    span = span_recorder.one("transcript.find_audio_url")
    assert span.attributes["transcript.matched"] is False


def test_download_emits_span_with_bytes_and_duration(
    span_recorder,  # type: ignore[no-untyped-def]
) -> None:
    fake_factory = _build_fake_factory_that_writes_audio()
    source = YouTubeSource(factory=fake_factory, verify_yt_dlp=False)
    data = source.download("https://example/watch")
    span = span_recorder.one("transcript.download")
    assert span.attributes["transcript.source_name"] == "youtube"
    assert span.attributes["transcript.bytes"] == len(data)
    assert span.attributes["transcript.duration_ms"] >= 0
```

Reuse the existing fake-yt-dlp factory pattern present in the file —
the helper names above are illustrative; substitute with the actual
test scaffolding used today.

- [ ] **Step 6.2: Run, confirm failure**

```bash
uv run --active pytest tests/unit/transcripts/test_youtube.py -v
```

Expected: 3 new tests FAIL.

- [ ] **Step 6.3: Implement**

In `src/auto_research/ingest/transcripts/sources/youtube.py`:

1. Add tracer:

```python
from opentelemetry import trace

_tracer = trace.get_tracer(__name__)
```

2. Wrap `find_audio_url`:

```python
def find_audio_url(self, ticker: str, year: int, quarter: int) -> str | None:
    """... existing docstring ..."""
    ticker_upper = ticker.upper()
    company = TICKER_QUERIES.get(ticker_upper, ticker_upper)
    query = f"{company} earnings call Q{quarter} {year}"

    with _tracer.start_as_current_span("transcript.find_audio_url") as span:
        span.set_attribute("transcript.ticker", ticker_upper)
        span.set_attribute("transcript.year", year)
        span.set_attribute("transcript.quarter", quarter)
        span.set_attribute("transcript.query", query)

        self.rate_limiter.wait()
        opts = {...}  # unchanged
        with self._factory(opts) as ydl:
            info = ydl.extract_info(
                f"ytsearch{self._search_limit}:{query}", download=False
            )

        entries = (info or {}).get("entries") or []
        span.set_attribute("transcript.result_count", len(entries))
        # ... existing entry-filtering loop ...
        # At the success-return site:
        #     span.set_attribute("transcript.matched", True)
        #     return url
        # At the fall-through (no match found):
        #     span.set_attribute("transcript.matched", False)
        #     return None
```

3. Wrap `download`. Capture wall-clock duration around the actual
yt-dlp invocation:

```python
def download(self, audio_url: str) -> bytes:
    """... existing docstring ..."""
    self.rate_limiter.wait()
    with _tracer.start_as_current_span("transcript.download") as span:
        span.set_attribute("transcript.source_name", SOURCE_NAME)
        start_ns = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="yt-") as tmpdir:
            # ... unchanged body ...
            audio = audio_file.read_bytes()
            # ... existing magic-byte + size checks ...
            span.set_attribute("transcript.bytes", len(audio))
            span.set_attribute(
                "transcript.duration_ms",
                (time.perf_counter_ns() - start_ns) // 1_000_000,
            )
            return audio
```

Add `import time` to the imports if not present.

- [ ] **Step 6.4: Run, confirm pass**

```bash
uv run --active pytest tests/unit/transcripts/test_youtube.py -v
```

Expected: PASS.

- [ ] **Step 6.5: Commit**

```bash
git add src/auto_research/ingest/transcripts/sources/youtube.py tests/unit/transcripts/test_youtube.py
git commit -m "feat(transcripts/youtube): emit find_audio_url + download spans

Refs #52."
```

---

## Task 7 — Instrument `DirectMp3Source.download`

**Files:**
- Modify: `src/auto_research/ingest/transcripts/sources/direct_mp3.py:1`
  + `.download` method.
- Modify: `tests/unit/transcripts/test_direct_mp3.py:1`.

- [ ] **Step 7.1: Write failing test**

```python
def test_download_emits_span(
    span_recorder,  # type: ignore[no-untyped-def]
) -> None:
    audio = b"ID3" + b"\x00" * 2048  # passes magic-byte + size floor
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=audio)
    )
    source = DirectMp3Source(transport=transport)
    result = source.download("https://example/audio.mp3")
    assert result == audio

    span = span_recorder.one("transcript.download")
    assert span.attributes["transcript.source_name"] == "direct_mp3"
    assert span.attributes["transcript.bytes"] == len(audio)
    assert span.attributes["transcript.duration_ms"] >= 0
```

- [ ] **Step 7.2: Run, confirm failure**

```bash
uv run --active pytest tests/unit/transcripts/test_direct_mp3.py -v
```

- [ ] **Step 7.3: Implement**

In `direct_mp3.py`:

```python
from opentelemetry import trace

_tracer = trace.get_tracer(__name__)
```

```python
def download(self, audio_url: str) -> bytes:
    """... existing docstring ..."""
    import time
    from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

    retryable = _http.retryable_exceptions(...)
    with _tracer.start_as_current_span("transcript.download") as span:
        span.set_attribute("transcript.source_name", SOURCE_NAME)
        start_ns = time.perf_counter_ns()
        for attempt in Retrying(...):
            with attempt:
                self.rate_limiter.wait()
                resp = self._client.get(audio_url)
                _http.classify_response(...)
                resp.raise_for_status()
                content = resp.content
                span.set_attribute("transcript.bytes", len(content))
                span.set_attribute(
                    "transcript.duration_ms",
                    (time.perf_counter_ns() - start_ns) // 1_000_000,
                )
                return content
        raise RuntimeError("unreachable: tenacity always returns or raises")
```

- [ ] **Step 7.4: Run, confirm pass**

```bash
uv run --active pytest tests/unit/transcripts/test_direct_mp3.py -v
```

- [ ] **Step 7.5: Commit**

```bash
git add src/auto_research/ingest/transcripts/sources/direct_mp3.py tests/unit/transcripts/test_direct_mp3.py
git commit -m "feat(transcripts/direct_mp3): emit transcript.download span

Refs #52."
```

---

## Task 8 — Instrument `extract_s_filing`

**Files:**
- Modify: `src/auto_research/extract/workers/s_filings.py:30` (imports) and
  `:188` (`extract_s_filing` body).
- Modify: `tests/unit/test_extract_worker_s_filings.py:1`.

- [ ] **Step 8.1: Write failing tests**

Append to `tests/unit/test_extract_worker_s_filings.py`:

```python
def test_extract_s_filing_emits_span_cache_hit(
    span_recorder, tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    """Pre-seeded cache → outcome=cache_hit, no LLM call."""
    # Use existing fixture(s) to seed the content cache.
    _seed_cache(...)
    extract_s_filing(
        raw_doc="...",
        doc_id="NVDA-S-3-2024",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
    )
    span = span_recorder.one("extract.s_filings")
    assert span.attributes["extract.worker"] == "s_filings"
    assert span.attributes["extract.doc_id"] == "NVDA-S-3-2024"
    assert span.attributes["extract.outcome"] == "cache_hit"


def test_extract_s_filing_emits_span_persisted(
    span_recorder, tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    """Successful extraction → outcome=persisted (parented over the
    client's llm.cost.est_usd span)."""
    mock_client = _build_mock_anthropic_returning_valid_dilution()
    extract_s_filing(
        raw_doc=_FIXTURE_S3_TEXT,
        doc_id="NVDA-S-3-2024",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
        anthropic_client=mock_client,
    )
    span = span_recorder.one("extract.s_filings")
    assert span.attributes["extract.outcome"] == "persisted"


def test_extract_s_filing_emits_span_quarantined(
    span_recorder, tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    """JSON-decode failure → outcome=quarantined."""
    mock_client = _build_mock_anthropic_returning_invalid_json()
    extract_s_filing(
        raw_doc=_FIXTURE_S3_TEXT,
        doc_id="NVDA-S-3-2024",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "quar",
        anthropic_client=mock_client,
    )
    span = span_recorder.one("extract.s_filings")
    assert span.attributes["extract.outcome"] == "quarantined"
```

Reuse the mock-Anthropic and fixture patterns already present in the
file.

- [ ] **Step 8.2: Run, confirm failure**

```bash
uv run --active pytest tests/unit/test_extract_worker_s_filings.py -v
```

- [ ] **Step 8.3: Implement**

In `src/auto_research/extract/workers/s_filings.py`:

1. Add tracer:

```python
from opentelemetry import trace

_tracer = trace.get_tracer(__name__)
```

2. Wrap the body of `extract_s_filing` and emit `extract.outcome`
on each return:

```python
def extract_s_filing(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> SFilingOutput | None:
    """... existing docstring ..."""
    with _tracer.start_as_current_span("extract.s_filings") as span:
        span.set_attribute("extract.worker", _WORKER)
        span.set_attribute("extract.doc_id", doc_id)

        # ... existing body up to cache check ...
        cached = content_cache.read(effective_cache_root, _WORKER, key)
        if cached is not None:
            span.set_attribute("extract.outcome", "cache_hit")
            return SFilingOutput.model_validate(cached)

        # ... existing body. On each quarantine path, set:
        #     span.set_attribute("extract.outcome", "quarantined")
        #     return None
        # On the final success path:
        #     content_cache.write(...)
        #     span.set_attribute("extract.outcome", "persisted")
        #     return validated
```

Apply the outcome attribute on every existing exit point (each
quarantine branch + the success path). Use a constant or local
variable if the branch count makes it noisy — but the spec sketch
above shows the four exits in this function.

- [ ] **Step 8.4: Run, confirm pass**

```bash
uv run --active pytest tests/unit/test_extract_worker_s_filings.py -v
```

- [ ] **Step 8.5: Commit**

```bash
git add src/auto_research/extract/workers/s_filings.py tests/unit/test_extract_worker_s_filings.py
git commit -m "feat(extract/s_filings): emit extract.s_filings parent span

Refs #52. The existing llm.cost.est_usd attribute set by
extract/client.py:151 now lands under a named parent."
```

---

## Task 9 — `docs/ARCHITECTURE.md` Observability subsection

**Files:**
- Modify: `docs/ARCHITECTURE.md` — add a dedicated subsection between
  the existing "External services" table (§6) and "Where to look
  when…" (§7).

- [ ] **Step 9.1: Add the subsection**

Replace the section break between §6 and §7 with a new `## 7.
Observability` section (renumbering subsequent sections):

```markdown
## 7. Observability

The strategy is ratified in `docs/specs/2026-05-22-design.md` §15:
one tracing backend (Langfuse via OTLP) for LLM-touching code, one
experiment store (MLflow) for backtests/signals, DuckDB notebooks
for ad-hoc analysis, no infra metrics layer (Prometheus/Grafana is
out of scope by design).

**Single init point.** Every process that does I/O calls
`auto_research.telemetry.try_init_telemetry()` at start. The
strict variant `init_telemetry()` is for tests and the integration
smoke; the CLI uses the env-tolerant wrapper.

**Entry-point catalog** (where init lives today):

| Entry point | Init call site |
|---|---|
| `auto-research ingest edgar` | `cli.py:ingest_edgar` |
| `auto-research extract s-filings` | `cli.py:extract_s_filings` |
| Integration tests under `tests/integration/` | `init_telemetry()` (strict) |
| Future: nightly batch worker (#19), live critic | TBD — call `try_init_telemetry()` at process start |

**Manual-span boundaries.** Auto-instrumentation (OpenLLMetry)
covers Anthropic / Whisper / future LangChain SDK calls. Manual
spans only at orchestration boundaries where parent/child grouping
matters:

| Span | Emitter |
|---|---|
| `edgar.fetch_filings_for_cik` | `ingest/edgar.py` |
| `transcript.fetch` | `ingest/transcripts/__init__.py` |
| `transcript.find_audio_url` | `ingest/transcripts/sources/youtube.py` |
| `transcript.download` | `youtube.py` + `direct_mp3.py` |
| `extract.s_filings` | `extract/workers/s_filings.py` |

LLM cost (`llm.cost.est_usd`) is set on the active span by
`extract/client.py:151` — workers do not duplicate this.

**No-op safety.** If telemetry is not initialized in a process,
`get_current_span()` returns the default no-op span and
`start_as_current_span` is a cheap pass-through; nothing crashes.
Spans simply do not get exported.
```

- [ ] **Step 9.2: Verify markdown renders**

```bash
uv run --active python -c "import pathlib; print(pathlib.Path('docs/ARCHITECTURE.md').read_text().count('## '))"
```

Expected: the section count increased by 1.

- [ ] **Step 9.3: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs(architecture): document observability strategy + entry points

Refs #52. Adds a dedicated subsection covering the single init
point, entry-point catalog, manual-span boundaries, and the
no-op safety property."
```

---

## Task 10 — Polish issue #52 body

**Files:** GitHub issue #52.

- [ ] **Step 10.1: Edit issue body**

```bash
gh issue edit 52 --repo investlink-ai/auto-research --body "$(cat <<'EOF'
## Context

The repo has working OTel infrastructure in `src/auto_research/telemetry.py`
(Traceloop → Langfuse OTLP export, tested in `test_telemetry.py` +
`test_telemetry_export.py`), but **no production code calls
`init_telemetry()` or emits manual spans**. Verified via grep: zero hits
in `src/` and `scripts/` for `get_tracer`, `start_as_current_span`,
`Traceloop`, `@workflow`, `@task` outside the telemetry module.

One worker-side hook already exists: `extract/client.py:151` sets
`llm.cost.est_usd` on the active OTel span (no-op-safe when no provider
is registered). Today that emission is dead in every real run because
no process initializes telemetry. This issue closes that loop.

The strategy this implements is ratified in
`docs/specs/2026-05-22-design.md` §15.

Surfaced during PR #49's review where the question "what about OTel
observability here?" caught the gap. Decided to land observability as
a separate, cross-cutting PR rather than retrofit one source.

## Objective

One coherent observability story across the ingest + extract layers:

1. **Wire `try_init_telemetry()` into entry points.** The natural home
   for W1 is the CLI module (`cli.py`'s `ingest edgar` + `extract
   s-filings`) — these are the W1 entry points that ship today. Future
   workers (W2 batch worker #19, live critic) wire the same call at
   their process-start sites. The strict `init_telemetry()` stays for
   tests / integration smoke.
2. **Emit manual spans for the orchestration layer:**
   - `transcript.fetch` around `fetch_transcript` — attrs: `transcript.ticker`,
     `.year`, `.quarter`, `.source_name`, `.outcome` (cached / unregistered /
     no_coverage / ok / error).
   - `transcript.find_audio_url` — attrs: query, result_count, matched.
   - `transcript.download` — attrs: source_name, bytes, duration_ms.
   - `edgar.fetch_filings_for_cik` — attrs: cik, form_types,
     n_filings, n_fetched, n_cache_hits.
   - `extract.s_filings` around `extract_s_filing` — attrs: worker,
     doc_id, outcome (cache_hit / persisted / quarantined). This
     parents the existing `llm.cost.est_usd` emission from
     `extract/client.py:151`.
3. **Take the OpenLLMetry free win.** Anthropic / Whisper SDK calls
   auto-trace once `init_telemetry()` runs in the process; no code
   change in `_whisper.py` or worker bodies needed.

## Acceptance criteria

- [ ] `auto-research ingest edgar` and `extract s-filings` call
      `try_init_telemetry()` at command start (verified by CliRunner
      tests that mock the helper).
- [ ] `fetch_transcript` emits a `transcript.fetch` span carrying the
      five outcomes (cached / unregistered / no_coverage / ok / error)
      — one assertion per branch.
- [ ] `fetch_filings_for_cik` emits an `edgar.fetch_filings_for_cik`
      span with the documented attributes.
- [ ] `YouTubeSource.find_audio_url` + `.download` and
      `DirectMp3Source.download` emit `transcript.find_audio_url` /
      `transcript.download` spans.
- [ ] `extract_s_filing` emits an `extract.s_filings` parent span;
      `extract.outcome` records cache_hit / persisted / quarantined.
- [ ] Unit-test coverage uses an in-memory `SpanRecorder` (per-test
      `InMemorySpanExporter` + `TracerProvider`); no live OTel
      exporter needed in CI. **Delivery to Langfuse** is already
      covered by the existing `tests/integration/test_telemetry_export.py`
      — this issue does not duplicate that work.
- [ ] `docs/ARCHITECTURE.md` gets a dedicated Observability
      subsection (single init point, entry-point catalog,
      manual-span boundaries, no-op safety).

## Blocked by

- Nothing. Decoupled from #19's batch worker — `cli.py` is the W1
  entry point that exists today, and adding init there now means
  future workers inherit the established pattern.
EOF
)"
```

- [ ] **Step 10.2: Verify**

```bash
gh issue view 52 --repo investlink-ai/auto-research | head -40
```

Expected: updated body visible.

(No commit — issue edit is out-of-repo.)

---

## Task 11 — Verify, commit residuals, open PR

**Files:** none directly; this is the wrap-up checklist.

- [ ] **Step 11.1: Run full quick gate**

```bash
make quick
```

Expected: ruff + mypy clean.

- [ ] **Step 11.2: Run full unit + feast suites**

```bash
uv run --active pytest tests/unit tests/feast -v
```

Expected: all green. Any pre-existing skips remain skipped; new tests
pass.

- [ ] **Step 11.3: Run telemetry integration smoke (if Langfuse is up)**

```bash
docker compose up -d
uv run --active pytest tests/integration/test_telemetry_export.py -v
docker compose down
```

Optional — gated on Docker availability. Don't block the PR on this if
the test environment can't run Docker; the unit-level SpanRecorder
tests are the CI gate.

- [ ] **Step 11.4: Push branch and open PR**

```bash
git push -u origin feat/52-otel-ingest-spans

gh pr create \
  --repo investlink-ai/auto-research \
  --title "feat(observability): cross-cutting OTel spans across ingest + extract (#52)" \
  --body "$(cat <<'EOF'
## Summary

Closes #52. Wires `try_init_telemetry()` into the W1 CLI commands and
emits manual OTel spans at five orchestration boundaries.

## Change

- `src/auto_research/telemetry.py` — add `try_init_telemetry()`
  best-effort wrapper around `init_telemetry()`.
- `src/auto_research/cli.py` — call `try_init_telemetry()` in
  `ingest edgar` and `extract s-filings`.
- `src/auto_research/ingest/edgar.py` — `edgar.fetch_filings_for_cik` span.
- `src/auto_research/ingest/transcripts/__init__.py` —
  `transcript.fetch` span with 5 outcomes.
- `src/auto_research/ingest/transcripts/sources/youtube.py` —
  `transcript.find_audio_url` + `transcript.download` spans.
- `src/auto_research/ingest/transcripts/sources/direct_mp3.py` —
  `transcript.download` span.
- `src/auto_research/extract/workers/s_filings.py` — `extract.s_filings`
  parent span (parents `extract/client.py:151`'s existing
  `llm.cost.est_usd` emission).
- `tests/conftest.py` — `span_recorder` fixture (in-memory exporter).
- `docs/ARCHITECTURE.md` — new Observability subsection.

## Verification

- `make quick` clean.
- Unit tests added per AC; all pass.
- `tests/integration/test_telemetry_export.py` (existing) confirms
  end-to-end delivery to Langfuse when run with Docker up; this PR
  does not change that test.

## AC checklist

- [x] CLI commands call `try_init_telemetry()`.
- [x] `transcript.fetch` covers 5 outcomes.
- [x] `edgar.fetch_filings_for_cik` span emitted.
- [x] `YouTubeSource` + `DirectMp3Source` spans emitted.
- [x] `extract.s_filings` parent span emitted; outcomes recorded.
- [x] Unit tests use `SpanRecorder` (in-memory).
- [x] `docs/ARCHITECTURE.md` updated.
EOF
)"
```

- [ ] **Step 11.5: Mark PR URL in the issue**

```bash
gh issue comment 52 --repo investlink-ai/auto-research \
  --body "PR open: <URL from gh pr create>"
```

---

## Self-review notes (skill-required)

- **Spec coverage:** every #52 AC bullet is addressed by Tasks 3-9; the
  spec §15 strategy is documented unchanged in Task 9.
- **Placeholder scan:** all code blocks contain complete code; the
  illustrative `_build_fake_*` helper names in Tasks 4-8 point at
  patterns that already exist in the corresponding test file — the
  implementing agent reuses or extends, not invents.
- **Type consistency:** attribute keys use one namespace per emitter
  (`edgar.*`, `transcript.*`, `extract.*`); span names use lower-dot
  per OTel conventions; outcome strings match across Task 5 and Task 8
  (cached/cache_hit are intentionally different — transcript vs.
  worker — and documented in the table at the top of this plan).
