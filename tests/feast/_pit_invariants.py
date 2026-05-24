"""Shared structural-PIT property assertions for FeatureView tests.

Defends the lag-1 trading-day cutoff invariant (AGENTS.md INV-1) via
``exchange_calendars`` directly, so the assertions don't share a code path
with ``next_trading_day_cutoff`` itself. Used by
``tests/feast/test_pit_properties.py`` today; future FeatureView property
tests (``ten_k_features``, ``transcript_features``, etc.) should call this
helper rather than re-implement the five checks per file.

See ``.claude/skills/pit-check/SKILL.md`` for why a self-comparison against
``next_trading_day_cutoff`` is forbidden (the property reduces to
``events.map(f) == events.map(f)``, true regardless of ``f``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
from hypothesis import strategies as st

_NYSE = xcals.get_calendar("XNYS")
_UTC = ZoneInfo("UTC")

# Wide range: covers pre/post-Juneteenth (added 2022), leap years, year
# boundaries, NYSE holiday catalog, and DST transitions. Clamp the upper
# bound with ~6 months of headroom against the calendar's last_session so
# cal.date_to_session doesn't raise for events near the bound (the calendar
# rebuilds with each xcals release and may shrink). Force-strip tz on
# last_session — current xcals returns naive but minor versions have
# fluctuated; min() of mixed-tz datetimes raises at module import time.
_RANGE_START: datetime = datetime(2015, 1, 1)
_CALENDAR_HEADROOM = timedelta(days=30 * 6)
_RANGE_END: datetime = min(
    datetime(2026, 12, 31, 23, 59, 59),
    _NYSE.last_session.to_pydatetime().replace(tzinfo=None) - _CALENDAR_HEADROOM,
)


def event_datetimes_strategy() -> st.SearchStrategy[pd.Series]:
    """Hypothesis strategy generating non-empty tz-aware UTC event_datetime
    Series for FeatureView property tests. Range is clamped against the
    NYSE calendar's last_session so cutoffs never fall past the bound.
    """
    single = st.datetimes(
        min_value=_RANGE_START,
        max_value=_RANGE_END,
        timezones=st.just(_UTC),
    )
    return st.lists(single, min_size=1, max_size=50).map(
        lambda xs: pd.Series(pd.to_datetime(list(xs), utc=True), name="event_datetime")
    )


def assert_pit_invariants(df: pd.DataFrame) -> None:
    """Assert the five structural PIT invariants on a materialized frame.

    ``df`` must be non-empty and contain tz-aware ``event_datetime`` and
    ``as_of_ts`` columns. Preconditions surface as named ``AssertionError``s
    rather than cryptic pandas/xcals errors from inside the loop.

    The five properties (each independently sufficient to catch a class of
    regression):

    1. Strict lag — ``as_of_ts > event_datetime`` (rules out identity cutoffs).
    2. Session validity — ``as_of_ts``'s NYSE-local date is an actual session.
    3. Close-exactness — ``as_of_ts`` equals that session's actual close
       (early-close days included).
    4. Date-strictly-after — the session date is strictly later than the
       NYSE-local date of ``event_datetime``.
    5. Minimality — the session is the FIRST session strictly after the event's
       NYSE-local date (catches regressions that skip 2+ sessions, which 1-4
       all allow).
    """
    # Preconditions — surface bad inputs with named errors instead of
    # cryptic KeyError / TypeError ('Cannot compare tz-naive...') from the loop.
    assert len(df) > 0, "assert_pit_invariants: empty materialized frame (no rows)"
    missing = {"event_datetime", "as_of_ts"} - set(df.columns)
    assert not missing, f"assert_pit_invariants: missing columns: {sorted(missing)}"
    for col in ("event_datetime", "as_of_ts"):
        assert isinstance(df[col].dtype, pd.DatetimeTZDtype), (
            f"assert_pit_invariants: {col!r} must be tz-aware "
            f"(got dtype={df[col].dtype!r})"
        )
    for event, as_of in zip(
        df["event_datetime"].to_list(), df["as_of_ts"].to_list(), strict=True
    ):
        # 1. Strictly-later: rules out identity / same-day cutoffs.
        assert as_of > event, f"as_of_ts {as_of} not strictly after event {event}"
        # 2. as_of_ts falls on an actual NYSE trading session.
        as_of_session = pd.Timestamp(as_of.tz_convert(_NYSE.tz).date())
        assert _NYSE.is_session(as_of_session), (
            f"as_of_ts {as_of} on non-session date {as_of_session.date()}"
        )
        # 3. as_of_ts equals that session's actual close (handles early closes).
        assert as_of == _NYSE.session_close(as_of_session), (
            f"as_of_ts {as_of} != session_close({as_of_session.date()})"
        )
        # 4. The session is strictly after the NYSE-local date of event.
        event_et_date = event.tz_convert(_NYSE.tz).date()
        assert as_of_session.date() > event_et_date, (
            f"as_of session {as_of_session.date()} not > event ET date {event_et_date}"
        )
        # 5. Minimality: as_of_session is the FIRST trading session strictly
        # after the event's NYSE-local date. Uses sessions_in_range (a
        # different xcals API than date_to_session(direction="next")) for
        # independent verification.
        later_sessions = _NYSE.sessions_in_range(
            pd.Timestamp(event_et_date) + pd.Timedelta(days=1),
            pd.Timestamp(event_et_date) + pd.Timedelta(days=30),
        )
        assert as_of_session == later_sessions[0], (
            f"as_of_session {as_of_session.date()} is not the first session "
            f"strictly after event ET date {event_et_date} (got {later_sessions[0].date()})"
        )
