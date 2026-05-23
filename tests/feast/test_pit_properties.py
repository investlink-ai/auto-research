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

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from hypothesis import given, settings

from feast_repo._materialize import materialize_price_features
from feast_repo._pit import next_trading_day_cutoff
from tests.feast._pit_invariants import assert_pit_invariants, event_datetimes_strategy

_UTC = ZoneInfo("UTC")
_ET = ZoneInfo("America/New_York")
_NYSE_REGULAR_CLOSE_HOUR_ET = 16  # 4pm Eastern


def _price_features_events_frame(events: pd.Series) -> pd.DataFrame:
    """price_features-specific frame builder. Each FeatureView owns its own
    builder; the shared `assert_pit_invariants` and `event_datetimes_strategy`
    work across all of them.
    """
    n = len(events)
    return pd.DataFrame(
        {
            "entity_id": ["NVDA"] * n,
            # Pass the Series directly (not .to_numpy()) so the column dtype
            # stays DatetimeTZDtype even for n == 0 — otherwise the array
            # round-trip degrades to object and the materializer's tz check
            # fires for the wrong reason if min_size is ever relaxed.
            "event_datetime": events.reset_index(drop=True),
            "close_adj": [0.0] * n,
            "returns_1d": [0.0] * n,
            "returns_5d": [0.0] * n,
            "vol_20d_annualized": [0.0] * n,
            "bid_ask_half_spread_bps": [0.0] * n,
            "adv_20d_usd": [0.0] * n,
        }
    )


# ---- Structural PIT properties (independent of the function under test) ----


@given(events=event_datetimes_strategy())
@settings(max_examples=200, deadline=None)
def test_price_features_pit_invariants(events: pd.Series) -> None:
    """Hypothesis sweep: the five structural PIT invariants hold for every
    materialized row across a wide UTC date range. See
    :mod:`tests.feast._pit_invariants` for what's checked and why.
    """
    assert_pit_invariants(materialize_price_features(_price_features_events_frame(events)))


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


def _zero_features(n: int) -> dict[str, list[float]]:
    return {
        "close_adj": [0.0] * n,
        "returns_1d": [0.0] * n,
        "returns_5d": [0.0] * n,
        "vol_20d_annualized": [0.0] * n,
        "bid_ask_half_spread_bps": [0.0] * n,
        "adv_20d_usd": [0.0] * n,
    }


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
            **_zero_features(n),
        }
    )
    with pytest.raises(TypeError, match="tz-aware"):
        materialize_price_features(frame)


def test_materialize_price_features_rejects_nat_column() -> None:
    """A NaT row in event_datetime should be rejected at the boundary with the
    same typed error class as the per-row cutoff, so producer-side try/excepts
    catch both paths. docs/DATA_MODEL.md §1.1 documents `TypeError`.
    """
    n = 2
    ts_col = pd.to_datetime(["2024-06-04 10:00"], utc=True).append(
        pd.DatetimeIndex([pd.NaT], tz="UTC")
    )
    frame = pd.DataFrame(
        {
            "entity_id": ["NVDA"] * n,
            "event_datetime": ts_col,
            **_zero_features(n),
        }
    )
    with pytest.raises(TypeError, match="NaT"):
        materialize_price_features(frame)


def test_next_trading_day_cutoff_past_calendar_bound_reraises_typed() -> None:
    """Events past the NYSE calendar's `last_session` should surface a typed
    ValueError naming the calendar bound, not a cryptic xcals
    `DateOutOfBounds` from eleven frames deep.
    """
    too_far = pd.Timestamp(datetime(2099, 1, 1, 10, 0, tzinfo=_ET)).tz_convert(_UTC)
    with pytest.raises(ValueError, match="last_session"):
        next_trading_day_cutoff(too_far)


# ---- assert_pit_invariants preconditions (helper is intended for 3+ future FVs) ----


def _aware_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_datetime": pd.to_datetime(["2024-06-04 10:00"], utc=True),
            "as_of_ts": pd.to_datetime(["2024-06-05 20:00"], utc=True),
        }
    )


def test_assert_pit_invariants_rejects_empty_frame() -> None:
    empty = pd.DataFrame(
        {
            "event_datetime": pd.to_datetime([], utc=True),
            "as_of_ts": pd.to_datetime([], utc=True),
        }
    )
    with pytest.raises(AssertionError, match="empty"):
        assert_pit_invariants(empty)


def test_assert_pit_invariants_rejects_missing_columns() -> None:
    only_event = pd.DataFrame(
        {"event_datetime": pd.to_datetime(["2024-06-04 10:00"], utc=True)}
    )
    with pytest.raises(AssertionError, match="as_of_ts"):
        assert_pit_invariants(only_event)


def test_assert_pit_invariants_rejects_tz_naive_event_datetime() -> None:
    frame = _aware_row().assign(
        event_datetime=pd.to_datetime(["2024-06-04 10:00"])  # tz-naive on purpose
    )
    with pytest.raises(AssertionError, match="tz-aware"):
        assert_pit_invariants(frame)


# ---- Parquet round-trip: the actual production trigger of the naive bug ----


def test_naive_event_datetime_after_parquet_roundtrip_is_rejected(tmp_path: Path) -> None:
    """A naive event_datetime column stays naive through pyarrow parquet
    round-trip; the materializer must reject the degraded frame at the
    boundary instead of silently bucketing every row by UTC date.
    """
    n = 3
    naive = pd.DataFrame(
        {
            "entity_id": ["NVDA"] * n,
            # tz-naive — what some pyarrow writer/schema combinations produce.
            "event_datetime": pd.to_datetime(["2024-06-04 10:00"] * n),
            **_zero_features(n),
        }
    )
    path = tmp_path / "events.parquet"
    naive.to_parquet(path, index=False)
    roundtripped = pd.read_parquet(path)
    # Sanity: parquet round-trip preserved the (broken) naive dtype.
    assert not isinstance(roundtripped["event_datetime"].dtype, pd.DatetimeTZDtype)
    with pytest.raises(TypeError, match="tz-aware"):
        materialize_price_features(roundtripped)


def test_tz_aware_event_datetime_survives_parquet_roundtrip(tmp_path: Path) -> None:
    """The happy path: a tz-aware event_datetime column round-trips through
    parquet without losing its dtype, so the materializer's boundary check
    accepts it and produces as_of_ts.
    """
    n = 3
    aware = pd.DataFrame(
        {
            "entity_id": ["NVDA"] * n,
            "event_datetime": pd.to_datetime(["2024-06-04 10:00"] * n, utc=True),
            **_zero_features(n),
        }
    )
    path = tmp_path / "events.parquet"
    aware.to_parquet(path, index=False)
    roundtripped = pd.read_parquet(path)
    assert isinstance(roundtripped["event_datetime"].dtype, pd.DatetimeTZDtype)
    result = materialize_price_features(roundtripped)
    assert "as_of_ts" in result.columns
    assert (result["as_of_ts"] > result["event_datetime"]).all()


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
        # DST spring forward (Sun Mar 10, 2024 02:00 EST -> 03:00 EDT). All
        # events that day bucket to ET date = Mar 10 -> cutoff = Mar 11 close
        # (4pm EDT = 20:00 UTC). Two UTC instants spanning the DST moment
        # both yield the same Monday close.
        # 06:30 UTC = 01:30 EST (before the spring-forward moment).
        (
            pd.Timestamp(datetime(2024, 3, 10, 6, 30, tzinfo=_UTC)),
            _close_ts(2024, 3, 11),
        ),
        # 07:30 UTC = 03:30 EDT (after the spring-forward moment).
        (
            pd.Timestamp(datetime(2024, 3, 10, 7, 30, tzinfo=_UTC)),
            _close_ts(2024, 3, 11),
        ),
        # DST fall back (Sun Nov 3, 2024 02:00 EDT -> 01:00 EST). All events
        # that day bucket to ET date = Nov 3 -> cutoff = Nov 4 close (4pm EST
        # = 21:00 UTC). The 01:00-02:00 ET hour occurs twice; both instants
        # bucket to Nov 3.
        # 05:30 UTC = 01:30 EDT (the first 01:30 ET of the day).
        (
            pd.Timestamp(datetime(2024, 11, 3, 5, 30, tzinfo=_UTC)),
            _close_ts(2024, 11, 4),
        ),
        # 06:30 UTC = 01:30 EST (the second 01:30 ET of the day).
        (
            pd.Timestamp(datetime(2024, 11, 3, 6, 30, tzinfo=_UTC)),
            _close_ts(2024, 11, 4),
        ),
    ],
)
def test_next_trading_day_cutoff_holiday_anchors(
    event: pd.Timestamp, expected_close: pd.Timestamp
) -> None:
    assert next_trading_day_cutoff(event) == expected_close


def test_midnight_et_boundary_cliff_is_intentional() -> None:
    """Two events 2 seconds apart spanning midnight ET get cutoffs ONE
    trading day apart by design — the algorithm bins by NYSE-local date.

    A 23:59:59 ET event was published on Tuesday in NY market terms; a
    00:00:01 ET event was Wednesday. This is the conservative reading of
    "next trading day cutoff" and is the failure mode the property test's
    "minimality" invariant pins. Future maintainers should not "fix" this
    by switching to a wall-clock-delta semantics — Codex flagged it on
    PR #38 and the resolution was to document the intent, not change it.
    """
    just_before = pd.Timestamp(datetime(2024, 6, 4, 23, 59, 59, tzinfo=_ET))
    just_after = pd.Timestamp(datetime(2024, 6, 5, 0, 0, 1, tzinfo=_ET))
    # The two inputs are 2 real-time seconds apart...
    assert (just_after.tz_convert(_UTC) - just_before.tz_convert(_UTC)) == pd.Timedelta(
        seconds=2
    )
    cutoff_before = next_trading_day_cutoff(just_before)
    cutoff_after = next_trading_day_cutoff(just_after)
    # ...but their PIT cutoffs differ by ONE trading session (Wed close vs
    # Thu close), because ET date Jun 4 -> Jun 5 close and ET date Jun 5 ->
    # Jun 6 close.
    assert cutoff_before == _close_ts(2024, 6, 5)
    assert cutoff_after == _close_ts(2024, 6, 6)
    assert (cutoff_after - cutoff_before) == pd.Timedelta(days=1)
