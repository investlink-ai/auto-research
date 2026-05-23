"""Lag-1 trading-day point-in-time cutoff (AGENTS.md INV-1).

Single producer of ``as_of_ts`` across the codebase. Callers populate
``as_of_ts`` at *write*-time by calling :func:`next_trading_day_cutoff` on the
underlying event timestamp. Query-time arithmetic on ``as_of_ts`` is forbidden
(the ``pit-check`` skill greps for the wrong shapes).

See ``docs/DATA_MODEL.md`` §1 for the contract.
"""

from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd

# Single module-level calendar instance. exchange_calendars already memoises
# get_calendar internally, so an extra lru_cache here would be redundant.
_NYSE = xcals.get_calendar("XNYS")


def next_trading_day_cutoff(t: pd.Timestamp) -> pd.Timestamp:
    """Return the NYSE close of the next trading session strictly after the
    NYSE-local calendar date of ``t``.

    Semantics (per ``docs/DATA_MODEL.md`` §1):

    * ``t`` MUST be a tz-aware ``pd.Timestamp``; tz-naive inputs are rejected
      with ``TypeError`` rather than silently re-interpreted as UTC (an ET
      producer would otherwise be off by one trading day).
    * ``pd.NaT`` is rejected with ``TypeError`` at this boundary instead of
      surfacing as a cryptic ``AttributeError`` deep inside ``exchange_calendars``.
    * The result is the actual session close on the next trading day after
      ``t``'s date in America/New_York. Early-close days (Christmas Eve, day
      after Thanksgiving) return the session's actual early close (1pm ET).
    * Closed days (holidays, weekends) are skipped.

    The returned timestamp is tz-aware (UTC).
    """
    if pd.isna(t):
        raise TypeError("next_trading_day_cutoff: event_datetime is NaT (missing)")
    ts = pd.Timestamp(t)  # naive ok: validator below rejects if tz-naive
    if ts.tzinfo is None:
        raise TypeError(
            "next_trading_day_cutoff: event_datetime must be tz-aware "
            "(received naive timestamp); ET-local times must be localised "
            "explicitly to avoid silent off-by-one ET-date bucketing"
        )
    # Compute the NYSE-local date so a 2am UTC event on Jan 2 isn't bucketed
    # into Jan 1 ET (and vice versa during DST gaps).
    et_date = ts.tz_convert(_NYSE.tz).date()
    # The session strictly after t's date.
    next_day = pd.Timestamp(et_date) + pd.Timedelta(days=1)  # naive ok: date arithmetic
    try:
        next_session = _NYSE.date_to_session(next_day.date().isoformat(), direction="next")
    except xcals.errors.DateOutOfBounds as exc:
        raise ValueError(
            f"next_trading_day_cutoff: event_datetime {ts.isoformat()} maps to a "
            f"next-session date past the NYSE calendar last_session "
            f"({_NYSE.last_session.date().isoformat()}); upgrade exchange_calendars "
            f"or clamp the input"
        ) from exc
    close = _NYSE.session_close(next_session)
    return pd.Timestamp(close).tz_convert("UTC")
