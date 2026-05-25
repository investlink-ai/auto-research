"""Shared pytest fixtures for the auto-research test suite.

`span_recorder` makes every manual span recorded by production code
visible to a unit test via an in-memory exporter — no live OTLP
required. `tests/integration/test_telemetry_export.py` covers the
real-Langfuse path.

Design notes:

1. OTel's `ProxyTracer` caches the real tracer on first
   `.start_as_current_span` call and never refreshes. Production
   modules grab `_tracer = trace.get_tracer(__name__)` at import
   time; if any test exercises a production span before the
   tracer-provider fixture activates, the proxy caches the default
   no-op tracer and subsequent `span_recorder` tests silently
   record zero spans. To prevent that, the session-scoped fixture
   is `autouse=True` so it installs the real provider before any
   test code executes.

2. `SynchronousMultiSpanProcessor._span_processors` is grow-only
   (no remove API). Calling `add_span_processor` per test would
   accumulate N shut-down processors over a long suite — every
   span emission would then iterate all N on_end hooks. Instead,
   we install ONE `SimpleSpanProcessor` and ONE
   `InMemorySpanExporter` at session scope, and the per-test
   fixture calls `exporter.clear()` to reset the captured-span
   list between tests.

The `SpanRecorder` class lives in `tests/_otel_helpers.py` so test
files can import it for type annotations — see that module's
docstring for why this conftest is excluded from mypy.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from tests._otel_helpers import SpanRecorder

# Session-wide singletons. The TracerProvider is installed before any
# test runs (autouse session fixture); the exporter is the one place
# spans are accumulated, and the function-scoped `span_recorder`
# fixture clears it between tests.
_SESSION_PROVIDER: TracerProvider | None = None
_SESSION_EXPORTER: InMemorySpanExporter | None = None


@pytest.fixture(scope="session", autouse=True)
def _session_tracer_provider() -> Iterator[TracerProvider]:
    """Install one TracerProvider + one SimpleSpanProcessor +
    one InMemorySpanExporter for the entire test session.

    Autouse so it runs before any test (and before any production
    module's lazy tracer-acquisition path executes). Restoration on
    teardown resets the global slot to whatever was previously set
    (typically OTel's default ProxyTracerProvider).
    """
    global _SESSION_PROVIDER, _SESSION_EXPORTER
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    previous = trace.get_tracer_provider()
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    _SESSION_PROVIDER = provider
    _SESSION_EXPORTER = exporter
    try:
        yield provider
    finally:
        _SESSION_PROVIDER = None
        _SESSION_EXPORTER = None
        provider.shutdown()
        trace._TRACER_PROVIDER = previous  # type: ignore[attr-defined]


@pytest.fixture
def span_recorder() -> Iterator[SpanRecorder]:
    """Per-test view of recorded spans. Clears the session exporter at
    setup so each test sees only its own emissions."""
    assert _SESSION_EXPORTER is not None, (
        "_session_tracer_provider must run before span_recorder is requested"
    )
    _SESSION_EXPORTER.clear()
    yield SpanRecorder(_SESSION_EXPORTER)
    # No teardown clear: the next test's setup will clear, and leaving
    # the spans available between yield and next-setup is convenient
    # for post-mortem assertions in test wrappers.
