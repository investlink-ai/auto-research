# AI Code Style

Agent-generation overlay for `auto-research`. Tells Claude Code and Cursor how
to produce code in this repo without over-building, restating safety policy, or
breaking PIT / citation-grounding / cost-model contracts.

Canonical rules live in:

- `AGENTS.md` §2 — invariants.
- `AGENTS.md` §3 — sensitive paths.
- `docs/AI_WORKFLOW.md` — tier classification + verification gates.
- `docs/specs/2026-05-22-design.md` — design intent.

When this file repeats a canonical rule, treat the canonical doc as source of
truth and this file as the agent-facing reminder.

---

## 1. Generation Contract

Before writing code, an agent must:

- Read the closest existing module + its tests.
- Classify the change with `docs/AI_WORKFLOW.md`.
- Identify which `AGENTS.md` §2 invariants the change can affect.
- Keep the patch inside the issue scope.
- Use tests first for behavior changes.
- Leave unrelated style debt alone unless it blocks the issue.

Generated code should look hand-maintained: typed, local, boring, consistent
with surrounding files.

---

## 2. Simplicity And Reuse

Prefer direct code until reuse is real.

Use direct code when:

- The behavior appears once.
- The rule is clearer inline than behind a helper.
- A helper name would just restate the implementation.
- The abstraction would hide an invariant from §2.

Reuse or extend when:

- Two current call sites would otherwise duplicate validation or conversion.
- The abstraction is stable domain vocabulary (e.g., `HypothesisType`,
  `InfoReport`, `BacktestReport`, `LiveCriticOutput`).
- The boundary is required for tests, persistence, or eval gates.

Do not add a framework, registry, plugin layer, base class, or generic service
on first use. Do not create "manager" objects that blend extraction, signals,
backtest, and agents responsibilities.

---

## 3. Abstraction Rules

An abstraction is allowed only when it does at least one of these:

- Removes real duplication across current call sites.
- Encodes a stable domain concept already named in the spec.
- Creates a boundary needed by tests or eval harness.
- Makes an invariant easier to enforce mechanically (citation grounding,
  triple-barrier label correctness, CPCV fold validation).

An abstraction is not allowed when it:

- Exists because "we might need it later" (DSPy, online Feast serving,
  Black-Litterman — all designed-for, not built).
- Turns a simple branch into a class hierarchy.
- Hides PIT timestamp math, citation post-validation, cost-model plumbing, or
  deflated-Sharpe correction.
- Makes tests assert mocks instead of real domain behavior.
- Lets an LLM call sneak into the trading-decision path (INV-3).

Prefer pure functions and frozen Pydantic models. Use Protocols only at real
runtime boundaries (LangGraph node interface, MCP tool registration, embedder
adapter for the Voyage/BGE fallback).

---

## 4. Library-First Investigation

Before writing custom logic for a non-trivial concern (retries, HTTP, parsing,
scheduling, observability, audio/video handling, format conversion, async
plumbing, etc.), check whether a reliable SDK or standard-library module
already does the work. **Don't rebuild the wheel.**

Custom logic that duplicates a library's surface adds traceback indirection,
drift risk on dependency upgrades, and a maintenance burden the team didn't
sign up for. Worse, the custom path will silently fall behind the library
(missed retry conditions, missed headers, missed corner cases) and the failure
mode is invisible until production.

The investigation step is short and concrete:

- For any concern the standard library or an already-pinned dep can handle,
  read the relevant SDK / module docs for ~5 minutes before writing code.
- If a library exists and is reasonable, use it.
- If you reject it, write one line in the PR body: "considered X, rejected
  because Y" (e.g., "considered httpx-cache, rejected because we need
  Parquet-backed manifest semantics, not on-disk HTTP cache").

Real lessons from this repo:

- The OpenAI SDK has built-in retries (408 / 409 / 429 / 5xx + connection
  errors), exponential backoff, and `Retry-After` honoring — configurable via
  `OpenAI(max_retries=N)`. **Don't** wrap `client.audio.transcriptions.create`
  in tenacity; that re-implements what the SDK already ships.
- `httpx` provides connection pooling, retries via transport, and timeout
  composition. **Don't** build a custom HTTP client around `urllib`.
- `tenacity` handles retry / backoff / jitter / `Retry-After` for paths the
  SDK doesn't own (EDGAR, IR audio hosts). **Don't** write a manual
  `for i in range(n): try: …; sleep(2**i)`.
- `subprocess.run(timeout=...)` handles process timeouts. **Don't** spawn a
  watchdog thread.
- `pyarrow.parquet` handles atomic Parquet write semantics with the right
  fsync discipline (we wrap it in `manifest.append`; we don't roll our own
  schema serialization).

The rule applies recursively: if a library exists and we already use a
heavier alternative, the lighter library probably belongs at the new call
site too. Consistency across the codebase beats local cleverness.

---

## 5. Comment And Docstring Discipline

Docstrings and comments describe **what the code does and the stable rationale
behind it**. They should remain accurate when the original PR has long since
merged and the author has moved on.

**Don't reference PR numbers, Issue numbers, ticket IDs, sprint names, or
review-bot findings in docstrings or comments.** These are ephemeral: a PR#
is meaningful only while the GitHub UI is open to the right repo; after
merge it rots into `git blame` territory. The next contributor reads the code
without that context — the docstring has to stand on its own.

```python
# ❌ BAD
"""Unit tests for the token-bucket rate limiter (Issue #5)."""

# Per PR #43 review, no_coverage rows are PERMANENT cache hits.

# Regression test for the Codex P2 finding on PR #34.

# ✅ GOOD
"""Unit tests for the token-bucket rate limiter."""

# `no_coverage` rows are permanent cache hits — the past doesn't change.

# Regression test: the universe loader must reject ticker entries
# missing a `sub_universe` field instead of silently defaulting to "".
```

The right home for ephemeral references is **commit messages, PR
descriptions, and ADRs under `docs/decisions/`**. The development trail lives
there; the code itself shouldn't carry it.

**The one exception**: a comment that explains a non-obvious constraint
introduced by a specific bug fix can name the symptom directly (preferred) or,
if absolutely necessary, link the incident — but the description of WHY the
constraint matters must stand on its own without the link.

```python
# ❌ "x must be > 0 (see PR #34)"
# ✅ "x must be > 0; x == 0 puts ffmpeg's segment muxer into a
#     1000-chunk-per-second loop and exhausts the OpenAI billing budget."
```

This rule applies to:

- Module docstrings
- Function / class / method docstrings
- Inline comments
- Test docstrings (the "what does this test prove" sentence)

It does NOT apply to:

- ADRs under `docs/decisions/` — those are dated, frozen artifacts that
  reasonably reference the discussion that produced them.
- Commit messages, PR bodies — those are the development trail by design.
- `CHANGELOG.md`-style files if/when we adopt them.

---

## 6. Research-Specific Anti-Patterns

These are the high-cost mistakes a coding agent will make in this repo if
unconstrained. Do not generate:

**Lookahead by query convention.** A FeatureView read that subtracts a day at
query time. PIT lag is baked at write-time (INV-1). If you find yourself
typing `as_of_ts - timedelta(days=1)` in a join, stop and re-read §6.3 of the
spec.

**Silent extraction retry on validation failure.** Citation-grounding failures
must route to `data/quarantine/`. Don't add a `try/except → log → return None`
that swallows the failure (INV-2).

**LLM as decision authority on promote/iterate/kill.** The gates are
code-checked constants. The LLM critic produces *qualitative addendum* text,
appended to the memo. It does not vote (INV-3).

**Naive Sharpe in any backtest output.** Use `deflated_sharpe` from
`backtest/deflated_sharpe.py`. Reporting `sharpe = mean(returns) / std(returns)`
in a `BacktestReport` field is invalid (INV-4).

**Gross-of-cost results.** Every backtest result that appears in a report,
memo, or signal-card must be net of the cost model (INV-5). If a notebook needs
gross for diagnosis, name the variable `sharpe_gross_diagnostic_only`.

**Write tools in the MCP server.** The MCP surface is read-only:
`query_features`, `run_backtest`, `search_memos`, `list_alpha_library`,
`read_signal_performance`, `get_feature_definition`. Don't add
`promote_signal`, `update_universe`, or `write_memo` — those mutate state and
the agent owns them through the graph, not through tool calls.

**Mocking the LLM in eval tests.** DeepEval and Ragas exist to test the real
extraction. Mocking returns canned outputs only for unit tests of pipeline
glue, never for citation-grounding tests or G-Eval scorers.

**Shadow state outside Feast for PIT-relevant features.** If a feature
participates in any signal, its authority is the FeatureView, period. No
side-channel DataFrames cached on an agent class, no "fast path" reads of raw
extraction JSON that bypass Feast (INV-1).

**Prompt edits without version bump.** Extraction is content-hash cached on
`(raw_doc, prompt_version)`. Editing prompt text without bumping its registry
version corrupts the cache contract and invalidates eval baselines (INV-6).

**Quiet defaults that mask absent data.** Missing transcripts on small
frontier-tech names are *real signal* about coverage, not a value to impute. Use
`null` + provenance metadata, then re-weight in the signal layer.

---

## 7. Sensitive-Code Generation Style

For Tier 2 work, the implementation should make the correctness case obvious:

- Inputs are explicit; no hidden defaults for decision-relevant values.
- Rejections are named (`ValueError("source_quote does not match span")`) and
  logged with context, never swallowed.
- Tests exercise the real domain object (`triple_barrier_label(...)`,
  `cpcv_split(...)`, `compute_deflated_sharpe(...)`), not a mock of it.
- Property-based tests (Hypothesis) for purged-fold non-overlap, embargo
  enforcement, and deflation monotonicity.
- LLM-touching code includes a `TestModel` (Pydantic AI) or `FakeListLLM`
  (LangChain) unit test path so the graph runs without an API call in CI.

Avoid clever compression in sensitive code. A few explicit branches with named
outcomes are better than a compact generic dispatcher that obscures behavior.

---

## 8. Pre-Submit Checklist

Before claiming a generated patch is ready:

- Does every changed file belong to the issue scope?
- Did behavior changes get a failing test first?
- Is every new abstraction justified by current duplication or a stable domain
  concept from the spec?
- For any non-trivial logic, did you check whether a reliable SDK / library
  already handles it (§4)? If you rejected one, is the rationale in the PR
  body?
- Do new docstrings and comments stand on their own without PR / Issue refs
  (§5)?
- Are docs synced without duplicating policy? (Link, don't restate.)
- For Tier 2, did the relevant `AGENTS.md` §2 invariant get an explicit test
  or eval citation in the PR body?
- For extraction work, did you bump `prompt_version` in the registry?
- For backtest work, is every reported Sharpe a deflated Sharpe, net of costs?
- For MCP-server work, is the new tool read-only?

If any answer is no, fix it before opening the PR.
