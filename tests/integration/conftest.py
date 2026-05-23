"""Integration-test shared setup.

- Auto-apply the `integration` marker to tests in this folder so individual
  files don't need `@pytest.mark.integration` decorators. The hook receives
  the session-wide item list, so we filter by path explicitly — otherwise
  unit tests collected in the same session would be stamped too.
- Probe Langfuse's health endpoint once per session; if down, skip tests
  that actually need Langfuse running. VCR-replay tests (EDGAR, FMP, …)
  don't talk to Langfuse and run regardless.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest


def _is_integration_item(item: pytest.Item) -> bool:
    """True iff the test lives under tests/integration/."""
    return "tests/integration" in str(item.path).replace("\\", "/")


def _needs_langfuse(item: pytest.Item) -> bool:
    """Tests whose filename contains `telemetry` talk to Langfuse.

    Filename-based via `Path.name` so backslash-separated paths on
    Windows behave the same as POSIX. New non-Langfuse integration
    tests just need a different filename.
    """
    return "telemetry" in Path(str(item.path)).name


def _langfuse_healthy(
    host: str = "localhost", port: int = 3000, timeout: float = 2.0
) -> bool:
    """GET Langfuse's documented health endpoint.

    Verifies it's Langfuse responding, not some other service squatting on
    :3000 (avoids the silent confusing-error case the reviewer flagged).
    """
    url = f"http://{host}:{port}/api/public/health"
    try:
        # urlopen raises HTTPError on 4xx/5xx; reaching the body means 2xx/3xx.
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark + conditionally-skip integration tests in one pass.

    Path-filtered so unit tests collected in the same run are untouched.
    Langfuse probe runs at most once per session and only when at least
    one Langfuse-needing test was collected.
    """
    integration_items = [item for item in items if _is_integration_item(item)]
    if not integration_items:
        return

    for item in integration_items:
        item.add_marker(pytest.mark.integration)

    langfuse_items = [item for item in integration_items if _needs_langfuse(item)]
    if langfuse_items and not _langfuse_healthy():
        skip = pytest.mark.skip(
            reason=(
                "Langfuse not reachable at localhost:3000/api/public/health. "
                "Start with: docker compose up -d"
            )
        )
        for item in langfuse_items:
            item.add_marker(skip)
