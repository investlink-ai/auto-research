"""Lag-1 trading-day point-in-time cutoff (AGENTS.md INV-1).

Single producer of ``as_of_ts`` across the codebase. Callers populate
``as_of_ts`` at *write*-time by calling :func:`next_trading_day_cutoff` on the
underlying event timestamp. Query-time arithmetic on ``as_of_ts`` is forbidden
(the ``pit-check`` skill greps for the wrong shapes).

See ``docs/DATA_MODEL.md`` §1 for the contract.
"""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=1)
def _nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS")


def next_trading_day_cutoff(t: pd.Timestamp) -> pd.Timestamp:
    """Return the NYSE close of the next trading session strictly after the
    NYSE-local calendar date of ``t``.

    Semantics (per ``docs/DATA_MODEL.md`` §1):

    * ``t`` is the publication / event timestamp (tz-aware; naive is treated as UTC).
    * The result is the actual session close on the next trading day after
      ``t``'s date in America/New_York. Early-close days (Christmas Eve, day
      after Thanksgiving) return the session's actual early close (1pm ET).
    * Closed days (holidays, weekends) are skipped.

    The returned timestamp is tz-aware (UTC).
    """
    cal = _nyse()
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    # Compute the NYSE-local date so a 2am UTC event on Jan 2 isn't bucketed
    # into Jan 1 ET (and vice versa during DST gaps).
    et_date = ts.tz_convert(cal.tz).date()
    # The session strictly after t's date.
    next_day = pd.Timestamp(et_date) + pd.Timedelta(days=1)
    next_session = cal.date_to_session(next_day.date().isoformat(), direction="next")
    close = cal.session_close(next_session)
    return pd.Timestamp(close).tz_convert("UTC")
