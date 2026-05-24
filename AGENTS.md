# auto-research — Agent Reference

Read this before any non-trivial work in this repo. Both Claude Code and Cursor
should treat it as the always-loaded system overview. `CLAUDE.md` and
`.cursor/rules/auto-research.mdc` are thin pointers; this file is the source.

Authority order:

1. Direct user instruction and the GitHub issue acceptance criteria.
2. This file (`AGENTS.md`), especially §2 invariants.
3. `docs/AI_WORKFLOW.md`, `docs/AI_CODE_STYLE.md`, `docs/ARCHITECTURE.md`,
   `docs/CONTRACTS.md`, `docs/DATA_MODEL.md`, `docs/BACKTEST.md`.
4. `docs/specs/2026-05-22-design.md` — frozen v1 design rationale and
   interview narrative; canonical docs above win on current behavior.
5. `docs/plans/` — task ordering, consumed in W1 then archived.
6. Tool-specific or third-party skills, prompts, memories, local habits.

Superpowers skills (`brainstorming`, `writing-plans`, `executing-plans`,
`verification-before-completion`, etc.) are allowed and actively used. When a
skill's process conflicts with this repo's workflow, the repo wins.

---

## 1. System Overview

`auto-research` is a two-plane multi-agent platform for cross-asset
language-driven alpha in AI infrastructure and frontier-tech equities.

**Where to read:**

- `docs/ARCHITECTURE.md` — current module map, plane boundaries, data flow.
- `docs/CONTRACTS.md` — Pydantic schemas, MCP tool surface, agent state.
- `docs/DATA_MODEL.md` — Feast FeatureViews, PIT contract, migrations.
- `docs/BACKTEST.md` — T1/T2 gates, CPCV, deflated Sharpe, cost model.
- `docs/specs/2026-05-22-design.md` — frozen v1 design + rationale + interview narrative.
- `docs/plans/2026-05-22-auto-research-implementation.md` — task ordering.

**Two planes** (full diagram and module map in `docs/ARCHITECTURE.md`):

- **Deterministic plane** — Feast PIT feature store, vbt.pro backtest engine,
  CPCV + deflated Sharpe + realistic cost model, paper portfolio.
- **LLM plane** — nightly batch extraction from SEC filings + earnings
  transcripts; LangGraph research agent; Pydantic AI live critic. The LLM
  never sits in the trading hot path.

The Feast feature store is the **only contract** between the two planes.

---

## 2. Non-Negotiable Invariants

**INV-1. PIT lag-1 is structural.** Every Feast row carries
`event_datetime` (publication time) and `as_of_ts = event_datetime + 1 trading
day cutoff`, **baked in at write-time**. Never enforced by query convention.
The PDF's "single most common silent killer" lookahead failure mode must remain
architecturally impossible.

**INV-2. Every extracted claim is citation-grounded.** Pydantic schemas in
`src/auto_research/extract/schemas.py` carry `source_span: tuple[int, int]` and
`source_quote: str` on every claim. Post-validation asserts
`source_text[span[0]:span[1]] == source_quote`. Failures route to
`data/quarantine/` for human review — never silently retried with degraded
data.

**INV-3. LLM never sits in the trading-decision path.** Extraction is nightly
batch with content-hash idempotent cache. Research is asynchronous. Live critic
emits a multiplicative haircut ∈ [0, 1] — never a directional override.
Promote/iterate/kill decisions in the research graph are **code-checked**
against `T1_GATE`/`T2_GATE` constants; the LLM critic produces only a
qualitative addendum to the memo.

**INV-4. Backtest discipline is López de Prado, not naive.** CV is CPCV with
embargo. Labels are triple-barrier. Sharpe is deflated across all signal
hypotheses + parameter sweeps. No vanilla k-fold, no naive Sharpe, no
in-sample evaluation reported as out-of-sample.

**INV-5. Costs are plumbed, not assumed.** FMP bid-ask half-spread + sqrt
impact + IBKR borrow proxy + commissions feed into `vbt.Portfolio.from_signals`.
Any backtest result must be reported net of costs. Gross-Sharpe-only claims
are invalid.

**INV-6. Determinism: completion configs are version-pinned.** Extraction
workers are
`(raw_doc, prompt_version, schema_version, model_id, decoding_params) → ExtractionOutput`
pure functions with content-hash idempotent cache
(`src/auto_research/extract/cache.py`). Prompt registry lives in Langfuse;
prompt and output-schema versions are colocated in code
(`extract/prompts/<name>.py` and `extract/schemas.py` `ClassVar`).
Changing any of the five inputs without invalidating the cache key
silently corrupts outputs. The `bump-prompt-version` skill defends
prompt + schema co-versioning; the cache key itself defends `model_id`
and `decoding_params`. Promotion to the Langfuse `production` tag is
gated by `scripts/promote_prompt.py` — eval-gated, not a manual flip.

**INV-7. Secrets never leak.** No agent or script reads `.env` directly, dumps
environment variables, or logs Anthropic / FMP / Voyage credentials. Diagnostics
may report presence/absence or masked values (`***`). `.claude/settings.json`
denies reads of `.env*`, `secrets/**`, `*.key`, `*.pem`.

---

## 3. Sensitive Paths

Edits to these paths require: a failing test first (or eval delta), an
explicit success criterion, and PR evidence that names the test. See
`docs/AI_WORKFLOW.md` for the 2-tier classification.
Marked `[SENSITIVE]` in `docs/ARCHITECTURE.md` §3.

- `src/auto_research/extract/guardrails.py` — citation-grounding validator
- `src/auto_research/extract/schemas.py` — `source_span` / `source_quote` contract
- `feast_repo/` (FeatureView definitions, `apply` configs) — PIT discipline
- `src/auto_research/backtest/cpcv.py` — combinatorial purged CV with embargo
- `src/auto_research/backtest/deflated_sharpe.py` — multiple-testing correction
- `src/auto_research/backtest/labels.py` — triple-barrier labels
- `src/auto_research/backtest/costs.py` — cost model plumbed into vbt
- `src/auto_research/backtest/gates.py` — `T1_GATE` / `T2_GATE` constants
- `src/auto_research/agents/research_graph.py` — graph state machine
- `src/auto_research/agents/reliability.py` — cost cap, circuit breaker, fallback
- `src/auto_research/mcp_server.py` — read-only tool surface (no write tools)

---

## 4. Working Style

Adapted from
[karpathy-skills/CLAUDE.md](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md);
folded in here so there is one policy source.

**Surface tradeoffs; don't pick silently.** If multiple interpretations exist,
present them. If a simpler approach exists, say so. If something is unclear,
stop and name what's confusing — especially around invariants in §2 (PIT,
citation grounding, CPCV, costs). Silent reasonable-looking choices are how
research codebases ship lookahead bugs.

**Surgical changes.** Every changed line should trace to the user's request or
the issue's acceptance criteria. Don't "improve" adjacent code, comments, or
formatting. Don't refactor things that aren't broken. Remove only orphans your
own changes created.

**Goal-driven, verify-then-loop.** Convert each task into a verifiable goal
before coding:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Implement signal A2" → "Pytest gauntlet hits `T1_GATE` thresholds end-to-end"
- "Tune chunking" → "Ragas `context_recall > 0.75` and `faithfulness > 0.85`"

For multi-step work, state the plan as `Step → verify` pairs. The verification
gate for sensitive paths is the corresponding test/eval; for ordinary code it's
`make quick` (ruff + mypy) + targeted unit test.

**Simplicity first.** Minimum code that solves the problem. No abstractions
for single-use code, no "flexibility" that wasn't requested, no error handling
for impossible scenarios. See `docs/AI_CODE_STYLE.md` for the full rules.

**Library-first.** Check for a reliable SDK or stdlib module before writing
custom logic for any non-trivial concern. See `docs/AI_CODE_STYLE.md` §4.

**Stable docstrings.** No PR/Issue numbers or ticket IDs in docstrings or
comments — that's commit-message territory. See `docs/AI_CODE_STYLE.md` §5.

---

## 5. Operational Prohibitions

Without explicit approval in the current task, agents must not:

- Push secrets, commit `.env`, or log credentials.
- Delete or rewrite files under `data/raw/`, `data/extracted/`, or
  `data/quarantine/` — they are the audit trail.
- Bypass the post-validation citation-grounding check, even temporarily.
- Run extraction backfill (~$75-150 in API spend) without user confirmation.
- Force-push to `main`, delete branches, or merge PRs.
- Mark Tier 2 PRs as "ready" when the citation-grounding / PIT / CPCV /
  deflated-Sharpe tests aren't passing.

---

## 6. Verification

| Tier | Gate |
|---|---|
| **0** Editorial (typos, comments, READMEs) | Diff inspection. |
| **1** Ordinary code | `make quick` (ruff + mypy) + targeted unit test. |
| **2** Sensitive paths (§3) | Failing test first, full pytest suite for the touched module, plus relevant eval (DeepEval for extraction, Ragas for RAG, CPCV for backtest). PR body cites the test name. |

Conventional-commit messages. Issue-driven branches (`feat/N-short-slug` from
`origin/main`, via worktree at `~/Documents/projects/auto-research/.worktree/N-slug/`).
PR body maps to acceptance criteria, not raw logs.

See `docs/AI_WORKFLOW.md` for the full workflow and PR evidence template.
