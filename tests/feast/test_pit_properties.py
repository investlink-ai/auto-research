"""Property tests for the lag-1 trading-day PIT cutoff (AGENTS.md INV-1).

The mandatory assertion (verified by the `pit-check` skill):

    as_of_ts == next_trading_day_cutoff(event_datetime)

holds for every row in every materialized FeatureView. If a future change
swaps the materializer to ``event_datetime + pd.Timedelta(days=1)`` or
similar fixed delta, this test fails.

See ``docs/DATA_MODEL.md`` §1 for the contract and ``AGENTS.md`` INV-1 for
the rationale (the "single most common silent killer" of research codebases).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from feast_repo._materialize import materialize_price_features
from feast_repo._pit import next_trading_day_cutoff

_UTC = ZoneInfo("UTC")
_ET = ZoneInfo("America/New_York")
_NYSE_REGULAR_CLOSE_HOUR_ET = 16  # 4pm Eastern

# Wide range: covers pre/post-Juneteenth (added 2022), leap years, year
# boundaries, NYSE holiday catalog, and DST transitions.
_RANGE_START = datetime(2015, 1, 1)
_RANGE_END = datetime(2026, 12, 31, 23, 59, 59)


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


# ---- The gate: the property assertion required by the issue AC + pit-check skill ----


@given(events=_event_datetimes())
@settings(max_examples=200, deadline=None)
def test_as_of_ts_is_next_trading_day_cutoff_for_price_features(events: pd.Series) -> None:
    df = materialize_price_features(_events_frame(events))
    expected = events.map(next_trading_day_cutoff).reset_index(drop=True)
    actual = df["as_of_ts"].reset_index(drop=True)
    assert (actual == expected).all()


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
