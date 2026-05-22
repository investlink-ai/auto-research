"""Integration-test shared setup.

- Auto-apply the `integration` marker to tests in this folder so individual
  files don't need `@pytest.mark.integration` decorators. The hook receives
  the session-wide item list, so we filter by path explicitly — otherwise
  unit tests collected in the same session would be stamped too.
- Probe Langfuse's health endpoint once per session; if down, mark every
  integration item as skipped with a useful message.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest


def _is_integration_item(item: pytest.Item) -> bool:
    """True iff the test lives under tests/integration/."""
    return "tests/integration" in str(item.path).replace("\\", "/")


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

    Single probe per session. Path-filtered so unit tests collected in
    the same run are untouched.
    """
    integration_items = [item for item in items if _is_integration_item(item)]
    if not integration_items:
        return

    for item in integration_items:
        item.add_marker(pytest.mark.integration)

    if not _langfuse_healthy():
        skip = pytest.mark.skip(
            reason=(
                "Langfuse not reachable at localhost:3000/api/public/health. "
                "Start with: docker compose up -d"
            )
        )
        for item in integration_items:
            item.add_marker(skip)
