---
name: chestertons-fence
description: Use BEFORE removing any decorator, config parameter, framework wiring, validation, exception handler, or method that existed in the original code (whether from a self-review finding, external reviewer feedback, or "this looks redundant" instinct). Forces an explicit understand-then-test-then-decide cycle so a wrong rationalisation doesn't silently regress behaviour. Catches the failure mode that hit PR #38: dropping `created_timestamp_column` on a Feast FileSource because the documented field semantic was "inverted" — which broke PIT tie-break determinism for intraday duplicates and shipped through a self-review batch before round-3 review caught it via reproduction.
allowed-tools: Read, Grep, Bash, Edit, Write
---

# chestertons-fence

> If you find a fence in a field, don't remove it until you understand why it
> was put there. When you DO understand, you may be right to remove it — but
> the understanding comes first.

Applied to code review: before removing anything that was present in the
original code, you must (1) understand WHY it was there, (2) write a
falsifying test FIRST, (3) remove the thing, (4) verify the falsifying test
now fails, (5) decide based on the failure.

The trap: a "this looks redundant / inverted / wrong" finding is
high-confidence in shape but low-confidence in CONSEQUENCE. The fence's
purpose may not match the framework's documented intent. Removing on
shape alone is how silent regressions ship.

## When to invoke

Before any commit that removes:

- A decorator (`@lru_cache`, `@retry`, `@property`, `@classmethod`).
- A keyword argument to a framework call (Feast `FileSource(...,
  created_timestamp_column=...)`, Pydantic `Field(...,  exclude=...)`,
  SQLAlchemy `Column(..., nullable=False)`, FastAPI `Depends(...)`).
- A method override, hook, or callback.
- A type annotation that affects runtime behaviour
  (`Annotated[T, Validator(...)]`).
- A previously-required argument now made optional.
- A validation step ("I already validate this earlier, this is redundant").
- An exception handler ("this can't happen").
- A test or assertion ("this is covered elsewhere").
- A config block in `pyproject.toml` / framework config files.

Don't invoke for: pure renames, formatting/whitespace, deleting orphaned
code with no callers anywhere, or removing things you added in the same
PR (those have no fence history yet).

## What "correct" looks like

A removal commit ends with three quoted artifacts: the WHY (history or
framework source), the falsifying test, and the test output in both
states.

```
fix(area): remove redundant @lru_cache around get_calendar

Chesterton's Fence check:

WHY (history): added in PR #12 as defensive memoisation against
repeated calendar construction in tight materializer loops.

WHAT framework actually does: exchange_calendars.get_calendar is
memoised at module level inside the library — verified by:
    >>> cal1 is xcals.get_calendar("XNYS")
    >>> cal2 is xcals.get_calendar("XNYS")
    >>> cal1 is cal2
    True

FALSIFYING TEST: tests/feast/test_pit_perf.py::test_cutoff_throughput
measures per-row cutoff cost over 10k calls. Quoted timing:
  - with @lru_cache:    median 1.92 µs/call
  - without @lru_cache: median 1.94 µs/call (Δ < 1%)

DECISION: safe to remove. The framework's own memoisation makes the
decorator load-zero, not load-bearing.
```

Three things make this commit safe to merge:

1. The "why" is grounded in repo history or framework docs/source, not a
   guess.
2. A test exists that distinguishes the two states (present vs absent).
3. The test was actually run; the result is quoted in the commit body.

## What "wrong" looks like

These rationalisations ship silent regressions:

- **"This looks redundant"** — without checking whether the framework
  treats the apparent duplication as redundant.
- **"The semantic is inverted vs the docs"** — the documented semantic
  describes intent; the actual behaviour may be what the original code
  wanted. Inversion of LABEL ≠ inversion of EFFECT. (PR #38's
  `created_timestamp_column='event_datetime'` was Feast's "ingestion
  time" field, semantically inverted because `event_datetime ≤ as_of_ts`
  always. But the BEHAVIOUR — Feast's `ORDER BY created_timestamp DESC`
  tie-break — produced the correct PIT-conservative reading. Removing it
  broke determinism for intraday duplicates.)
- **"No current callers exercise this"** — the absence of current
  callers doesn't tell you what FUTURE callers will need. Defensive
  wiring removed today bakes a regression into the next caller.
- **"The test suite passes with it removed"** — necessary but not
  sufficient. If no test exercises the protected behaviour, the suite
  proves nothing.
- **"I'll add a TODO to restore if we need it"** — by the time you need
  it, the bug has shipped and the silent failure has already corrupted
  data. Revert-as-safety-net doesn't work for silent bugs.

## Mandatory checks (run before claiming done)

Run all three. Stop and fix on any miss.

**1. Grep history for the introduction.** Find the commit that added the
thing, and read its message and diff:

```bash
git log -p --all -S '<the literal thing being removed>' -- '<path>' | head -200
git log --reverse --all --oneline -- '<path>' | head -5  # first commits touching the file
```

If the introduction commit's message explains the purpose, you have the
fence's reason. If it's a bare "initial scaffold" commit, read the
framework docs AND source to derive the purpose.

**2. Falsifying test exists and was run.** The test must:

- Currently PASS (because the thing is there).
- FAIL after removal (proving it catches the regression).
- (If you choose to keep the thing: PASS again after restoring.)

Run the cycle and quote the output of each state in the commit message.
If you can't write a test that distinguishes the two states, the thing
is probably load-zero — but say so explicitly in the commit message
("no behavioural difference observed; treating as cosmetic").

**3. Verify against framework SOURCE, not just docs.** Docs describe
intent; source describes effect. For Feast, Pydantic, FastAPI, SQLAlchemy
etc., read the relevant function in `.venv/lib/.../site-packages/<framework>`
or run a minimal reproduction. The PR-#38 case: Feast's offline-store
code at `feast/infra/offline_stores/ibis.py` showed
`ibis.desc(table[created_timestamp_col])` as the tie-break ORDER BY
clause — the docs called the field "ingestion timestamp" but the source
revealed it as a generic tie-breaker.

## Pre-submit checklist

- [ ] The introduction commit's reason is in scope (quoted in the new
      commit message, OR derived from framework source/docs and linked
      to the relevant lines).
- [ ] A falsifying test exists in this PR or a prior commit.
- [ ] The falsifying test was run with the thing present AND absent;
      both states' outputs are quoted in the commit message.
- [ ] The decision (remove / keep with new comment) is stated explicitly.
- [ ] If keeping the thing: the rationale comment in the CODE explains
      the non-obvious purpose so the next maintainer doesn't repeat the
      Chesterton's Fence cycle.
- [ ] If removing the thing: the falsifying test stays in the suite
      (with the inverted expectation or as a snapshot of current
      behaviour), so a future "let's add it back" change has the same
      gate.

## Escalation

If you removed something without this check and a later review caught a
silent regression (as happened with PR #38's `created_timestamp_column`):

1. **Restore the thing immediately.** Don't try to find an alternative
   path that "achieves the same thing without it" — that's how the
   original mistake compounds.
2. **Write the falsifying test you should have written before removal.**
   Pin the behaviour the removal broke.
3. **In the restore commit body, document the lesson.** Not just "fix
   regression" but: "I removed X because [wrong reason]; the actual
   purpose is [right reason from framework source]; falsifying test
   added at [path]." Future-you and future-reviewers need the trail.
4. **Consider whether the skill needs updating.** If the case was
   genuinely subtle (e.g. the framework's behaviour required reading
   non-obvious source files), add the framework or pattern to the
   "what 'wrong' looks like" examples here.

## Worked example: PR #38

The case that motivated this skill. Pasted verbatim so future readers
can see the failure mode in concrete form.

**The fence:** Original `feast_repo/feature_views.py` had
`FileSource(timestamp_field='as_of_ts', created_timestamp_column='event_datetime')`.

**Round-1 self-review rationalisation:** "Feast docs say
`created_timestamp_column` is the row's ingestion time, conventionally
≥ `timestamp_field`. But `event_datetime ≤ as_of_ts` always — this is
semantically inverted. The materializer emits one row per
`(entity_id, event_datetime)`, so there's no tie-breaker need today. Drop
the column."

**What that rationalisation missed:** The PIT join key is
`(entity_id, as_of_ts)`, not `(entity_id, event_datetime)`. Under the
lag-1 cutoff, ALL intraday events for one ticker on one ET trading day
collapse onto a single `as_of_ts` (the next session's close). So
multiple rows with identical `(entity_id, as_of_ts)` are not just
possible — they're the normal case under any real producer that emits
more than one intraday event per ticker per day. Feast's PIT join
without a tie-breaker becomes non-deterministic across pandas sort
stability / parquet row order.

**Round-3 review caught it via reproduction:** two AAPL rows at
2024-06-04 10:00Z and 15:30Z produced two parquet rows at identical
`as_of_ts=2024-06-05 20:00Z`. Verified Feast's offline-store source uses
`ORDER BY ... created_timestamp DESC` as tie-break.

**What the skill would have prevented:** step 2 (falsifying test before
removal) would have forced writing a test that emits two intraday events
and asserts a specific row wins. That test would have failed after
removal — preventing the bad commit from landing. Step 3 (read framework
source) would have shown the `created_timestamp_column` is generically
the tie-break key, and using `event_datetime` as the tie-break value is
a label/effect mismatch but a correct behaviour choice.
