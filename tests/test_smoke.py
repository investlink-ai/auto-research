"""Smoke test — verifies the package imports and exposes its version.

Every other test in this repo can assume the harness works because this
one passes.
"""

from auto_research import __version__


def test_version_matches_pyproject() -> None:
    assert __version__ == "0.1.0"
