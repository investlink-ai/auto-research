---
name: pit-check
description: Use when editing feast_repo/ or any code that writes a Feast FeatureView. Enforces INV-1 — the lag-1 trading-day point-in-time discipline must be baked at write-time, never enforced at query-time. The spec calls lookahead "the single most common silent killer"; this skill is the mechanical check that prevents it.
allowed-tools: Read, Grep, Bash
---

# pit-check

**Invariant being defended (AGENTS.md INV-1):**

> Every Feast row carries `event_datetime` (publication time) and
> `as_of_ts = event_datetime + 1 trading day cutoff`, **baked in at
> write-time**. Never enforced by query convention.

Without this, a single careless `get_historical_features` call leaks future
information into a backtest and invalidates the whole signal library. The
post-mortem cost of a lookahead bug is rerunning the 2-year backfill ($75-150)
plus losing the validation-discipline interview story.

## When to invoke

Invoke this skill before claiming done on any change that:

- Edits a file under `feast_repo/`
- Adds a new FeatureView or modifies an existing one
- Touches `feast_repo/_pit.py` (the `next_trading_day_cutoff` implementation)
- Adds code that writes to a Feast offline store
- Adds a backtest or signal that reads features via
  `fs.get_historical_features` and applies any filter on `as_of_ts`

## What "correct" looks like

Timestamps populated at write-time, inside the materializer:

```python
# feast_repo/feature_views.py — CORRECT
from feast_repo._pit import next_trading_day_cutoff

df["event_datetime"] = filings_df["accepted_datetime"]
df["as_of_ts"] = filings_df["accepted_datetime"].apply(next_trading_day_cutoff)
write_to_offline_store(df, ten_k_features)
```

Then reads are unadorned:

```python
# signals/a2_pead.py — CORRECT
features = fs.get_historical_features(
    entity_df=entity_df,
    features=fs.get_feature_service("signal_a2"),
).to_df()
# No further lag arithmetic. PIT join is automatic.
```

## What "wrong" looks like

Any of these patterns means INV-1 is being enforced at query-time, which
breaks the contract:

```python
# WRONG — query-time lag subtraction
features = features[features["as_of_ts"] <= entity_ts - pd.Timedelta(days=1)]

# WRONG — fixed timedelta instead of trading-day cutoff
df["as_of_ts"] = df["event_datetime"] + pd.Timedelta(days=1)  # ignores holidays/weekends

# WRONG — same-day as_of_ts (no lag)
df["as_of_ts"] = df["event_datetime"]

# WRONG — as_of_ts from a column other than event_datetime
df["as_of_ts"] = df["materialization_run_ts"]  # this is the trading-now, not data-available time

# WRONG — fillna with a permissive default
df["as_of_ts"] = df["as_of_ts"].fillna(pd.Timestamp.min)
```

## Mandatory checks (run before claiming done)

Run all four. Stop and fix on any miss.

**1. Grep for query-time lag patterns** (must return zero hits in
`src/auto_research/signals/`, `src/auto_research/backtest/`, and
`src/auto_research/agents/`):

```bash
rg -nP '(as_of_ts.*-.*Timedelta|as_of_ts\s*<\s*\w+_ts\s*-)' \
   src/auto_research/signals/ src/auto_research/backtest/ src/auto_research/agents/
```

**2. Grep for fixed-timedelta `as_of_ts` writes** (must return zero hits in
`feast_repo/`; the only correct constructor is `next_trading_day_cutoff`):

```bash
rg -nP "as_of_ts.*=.*Timedelta\(" feast_repo/
```

**3. Confirm `next_trading_day_cutoff` is the only producer of `as_of_ts`:**

```bash
rg -nP 'as_of_ts.*=' feast_repo/ src/auto_research/
# Every assignment should be either:
#   - df["as_of_ts"] = ...next_trading_day_cutoff(...)
#   - explicit test fixture in tests/
```

**4. Confirm a property test covers the invariant** for any new or modified
FeatureView. Required test (or equivalent):

```bash
rg -l 'as_of_ts.*event_datetime' tests/feast/
```

The property test in `tests/feast/test_pit_properties.py` must include this
shape (add a case if your new FeatureView isn't yet covered):

```python
@given(events=event_datetimes_strategy())
def test_as_of_ts_is_next_trading_day_cutoff(events):
    df = materialize_my_view(events)
    expected = events.apply(next_trading_day_cutoff)
    assert (df["as_of_ts"] == expected).all()
```

## Pre-submit checklist

- [ ] Every new write to a FeatureView populates `event_datetime` AND `as_of_ts`.
- [ ] `as_of_ts` is computed via `next_trading_day_cutoff(event_datetime)`,
      not a fixed `pd.Timedelta(days=1)`.
- [ ] No query-time arithmetic on `as_of_ts` in signals, backtest, or agents.
- [ ] Property test covers the new or modified FeatureView (grep #4).
- [ ] If holidays / weekends are involved, the `next_trading_day_cutoff`
      implementation is consulted — not reimplemented inline.
- [ ] PR body cites the test name (e.g.
      `tests/feast/test_pit_properties.py::test_as_of_ts_is_next_trading_day_cutoff_for_ten_k`).

## Escalation

If any of the above patterns appear in code that's already merged on `main`,
treat it as a P0 finding:

1. Stop further extraction or backtest work.
2. Open an issue labeled `bug` + `sensitive` + `pit`.
3. Identify which FeatureViews and which date ranges are corrupted.
4. Re-materialize the affected FeatureViews from `data/extracted/` after the
   fix lands. The raw store is intact; only the Feast offline store is
   corrupted.
5. Mark any backtest report that consumed corrupted features as invalid in
   MLflow (`mlflow.set_tag("invalid_due_to", "pit_bug_#N")`).

This is the recovery path because `data/raw/` and `data/extracted/` are
immutable (enforced by `.claude/settings.json` deny rule on `rm`). The Feast
offline store is a derived view and is always rebuildable.
