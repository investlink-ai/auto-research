"""Unit tests for the live-test opt-in tokenizer.

The auto-skip logic in `tests/live/conftest.py` uses `_is_live_opt_in`
to decide whether the user has explicitly asked to run live tests via
`-m`. A naive substring check (`"live" in markexpr`) false-positives
on `not live`, `alive`, `livedb`, etc. — these tests pin the
token-aware semantics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the live conftest as a standalone module (tests/ isn't a package).
_CONFTEST_PATH = Path(__file__).resolve().parents[1] / "live" / "conftest.py"
_spec = importlib.util.spec_from_file_location("live_conftest", _CONFTEST_PATH)
assert _spec is not None and _spec.loader is not None
_live_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_live_conftest)
_is_live_opt_in = _live_conftest._is_live_opt_in


@pytest.mark.parametrize(
    ("markexpr", "expected"),
    [
        # Explicit positive selections — opt-in.
        ("live", True),
        ("live or slow", True),
        ("slow or live", True),
        ("live and not deprecated", True),
        ("(live or canary)", True),
        # Negative selections — NOT opt-in.
        ("not live", False),
        ("not  live", False),
        ("slow and not live", False),
        # Substring traps the naive check would fail.
        ("alive", False),
        ("livedb", False),
        ("relive", False),
        ("delivery", False),
        # Empty / unset — not opt-in.
        ("", False),
        ("   ", False),
        # Unrelated markers — not opt-in.
        ("slow", False),
        ("integration and not eval", False),
    ],
)
def test_live_opt_in_recognizes_token_not_substring(markexpr: str, expected: bool) -> None:
    assert _is_live_opt_in(markexpr) is expected
