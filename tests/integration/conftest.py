"""Integration-test shared setup.

- Auto-apply the `integration` marker to every test in this folder so
  individual files don't need `@pytest.mark.integration` decorators.
- Session-scoped fixture verifies Langfuse is reachable; whole folder
  skips cleanly if not (typical when Docker isn't running locally).
"""

from __future__ import annotations

import socket

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply the `integration` marker to every test in this folder."""
    for item in items:
        item.add_marker(pytest.mark.integration)


def _langfuse_reachable(host: str = "localhost", port: int = 3000, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_langfuse_running() -> None:
    """Skip the whole integration folder if Langfuse isn't reachable on :3000."""
    if not _langfuse_reachable():
        pytest.skip(
            "Langfuse not reachable at localhost:3000. "
            "Start with: docker compose up -d",
            allow_module_level=True,
        )
