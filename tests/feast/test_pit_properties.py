"""Property tests for the lag-1 trading-day PIT cutoff (AGENTS.md INV-1).

The mandatory assertion (issue #7 AC + ``pit-check`` skill) is that every
materialized row carries ``as_of_ts == next_trading_day_cutoff(event_datetime)``.
But that bare equality is a tautology against an identity-perturbed cutoff
(both sides shift together), so this module pins a stronger set of structural
properties using ``exchange_calendars`` directly — independent of the function
under test:

* ``as_of_ts`` is strictly later than ``event_datetime`` (lag of at least one
  trading session).
* ``as_of_ts``'s NYSE-local date is a real NYSE trading session.
* ``as_of_ts`` equals that session's actual close (early-close days included).
* The session is strictly after the NYSE-local date of ``event_datetime``.
* ``next_trading_day_cutoff`` rejects tz-naive timestamps and ``pd.NaT`` with
  typed errors at the boundary (the producer cannot silently leak local-time
  or null timestamps past the materializer).

See ``docs/DATA_MODEL.md`` §1 and ``AGENTS.md`` INV-1 for the contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from feast_repo._materialize import materialize_price_features
from feast_repo._pit import next_trading_day_cutoff

_UTC = ZoneInfo("UTC")
_ET = ZoneInfo("America/New_York")
_NYSE_REGULAR_CLOSE_HOUR_ET = 16  # 4pm Eastern
_NYSE = xcals.get_calendar("XNYS")

# Wide range: covers pre/post-Juneteenth (added 2022), leap years, year
# boundaries, NYSE holiday catalog, and DST transitions. Clamp the upper
# bound with ~6 months of headroom against the calendar's last_session so
# cal.date_to_session doesn't raise for events near Dec 31 (the calendar
# rebuilds with each xcals release and may shrink, not just grow).
_RANGE_START = datetime(2015, 1, 1)
_CALENDAR_HEADROOM = timedelta(days=30 * 6)
_RANGE_END = min(
    datetime(2026, 12, 31, 23, 59, 59),
    _NYSE.last_session.to_pydatetime() - _CALENDAR_HEADROOM,
)


def _event_datetimes() -> st.SearchStrategy[pd.Series]:
    single = st.datetimes(
        min_value=_RANGE_START,
        max_value=_RANGE_END,
        timezones=st.just(_UTC),
    )
    return st.lists(single, min_size=1, max_size=50).map(
        lambda xs: pd.Series(pd.to_datetime(list(xs), utc=True), name="event_datetime")
    )


def _events_frame(events: pd.Series) -> pd.DataFrame:
    n = len(events)
    return pd.DataFrame(
        {
            "entity_id": ["NVDA"] * n,
            "event_datetime": events.to_numpy(),
            "close_adj": [0.0] * n,
            "returns_1d": [0.0] * n,
            "returns_5d": [0.0] * n,
            "vol_20d_annualized": [0.0] * n,
            "bid_ask_half_spread_bps": [0.0] * n,
            "adv_20d_usd": [0.0] * n,
        }
    )


# ---- Structural PIT properties (independent of the function under test) ----


@given(events=_event_datetimes())
@settings(max_examples=200, deadline=None)
def test_as_of_ts_is_next_trading_day_cutoff_for_price_features(events: pd.Series) -> None:
    df = materialize_price_features(_events_frame(events))
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


# ---- Boundary validation: producer cannot leak naive / NaT past the cutoff ----


def test_next_trading_day_cutoff_rejects_tz_naive() -> None:
    """Naive timestamps must not silently be treated as UTC — an upstream that
    writes naive ET wall-clock would otherwise get a one-trading-day-too-early
    cutoff (silent INV-1 violation).
    """
    with pytest.raises(TypeError, match="tz-aware"):
        next_trading_day_cutoff(pd.Timestamp(datetime(2024, 6, 4, 10, 0)))


def test_next_trading_day_cutoff_rejects_nat() -> None:
    """NaT must fail fast with a typed error instead of a cryptic AttributeError
    eleven frames deep inside exchange_calendars.
    """
    with pytest.raises(TypeError, match="NaT"):
        next_trading_day_cutoff(pd.NaT)  # type: ignore[arg-type]  # intentional


def test_materialize_price_features_rejects_tz_naive_column() -> None:
    """The materializer should reject a tz-naive event_datetime column at the
    boundary once, rather than letting next_trading_day_cutoff raise per-row.
    """
    n = 2
    frame = pd.DataFrame(
        {
            "entity_id": ["NVDA"] * n,
            # tz-naive on purpose — must be rejected.
            "event_datetime": pd.to_datetime(["2024-06-04 10:00", "2024-06-05 10:00"]),
            "close_adj": [0.0] * n,
            "returns_1d": [0.0] * n,
            "returns_5d": [0.0] * n,
            "vol_20d_annualized": [0.0] * n,
            "bid_ask_half_spread_bps": [0.0] * n,
            "adv_20d_usd": [0.0] * n,
        }
    )
    with pytest.raises(TypeError, match="tz-aware"):
        materialize_price_features(frame)


# ---- Holiday/weekend anchors: cheap explicit checks that document the contract ----


def _et_ts(year: int, month: int, day: int, hour: int = 10) -> pd.Timestamp:
    return pd.Timestamp(datetime(year, month, day, hour, tzinfo=_ET)).tz_convert(_UTC)


def _close_ts(year: int, month: int, day: int, hour: int = _NYSE_REGULAR_CLOSE_HOUR_ET) -> pd.Timestamp:
    return pd.Timestamp(datetime(year, month, day, hour, tzinfo=_ET)).tz_convert(_UTC)


@pytest.mark.parametrize(
    ("event", "expected_close"),
    [
        # Plain weekday: Tue 10am ET -> Wed close.
        (_et_ts(2024, 6, 4), _close_ts(2024, 6, 5)),
        # Friday -> Monday close.
        (_et_ts(2024, 6, 7), _close_ts(2024, 6, 10)),
        # Saturday -> Monday close.
        (_et_ts(2024, 6, 8), _close_ts(2024, 6, 10)),
        # Sunday -> Monday close.
        (_et_ts(2024, 6, 9), _close_ts(2024, 6, 10)),
        # Christmas Day 2024 (closed) -> Dec 26 close.
        (_et_ts(2024, 12, 25), _close_ts(2024, 12, 26)),
        # Day before Christmas (Tue Dec 24, NYSE early-close 1pm ET) is still
        # the trading day of `t`; cutoff is Dec 26 close.
        (_et_ts(2024, 12, 24), _close_ts(2024, 12, 26)),
        # Thanksgiving 2024 (Thu Nov 28, closed) -> Fri Nov 29.
        # Nov 29 itself is a NYSE early-close day (1pm ET).
        (_et_ts(2024, 11, 28), _close_ts(2024, 11, 29, hour=13)),
        # Juneteenth 2024 (Wed Jun 19, closed) -> Thu Jun 20.
        (_et_ts(2024, 6, 19), _close_ts(2024, 6, 20)),
        # New Year's Day 2024 (Mon Jan 1, closed) -> Tue Jan 2.
        (_et_ts(2024, 1, 1), _close_ts(2024, 1, 2)),
        # Good Friday 2024 (Fri Mar 29, closed) -> Mon Apr 1.
        (_et_ts(2024, 3, 29), _close_ts(2024, 4, 1)),
    ],
)
def test_next_trading_day_cutoff_holiday_anchors(
    event: pd.Timestamp, expected_close: pd.Timestamp
) -> None:
    assert next_trading_day_cutoff(event) == expected_close
