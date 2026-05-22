# AI Workflow

Workflow version: `auto-research:2026-05-22`

This is the repo-owned workflow for AI-assisted engineering in `auto-research`.
It applies equally to Claude Code and Cursor.

Authority order:

1. Direct user instruction and the GitHub issue acceptance criteria.
2. `AGENTS.md` §2 invariants.
3. This document, `docs/AI_CODE_STYLE.md`, `docs/specs/`, `docs/plans/`.
4. Tool-specific or third-party skills (superpowers, etc.) — allowed when
   compatible with the above; repo wins on conflict.

---

## 1. Start Every Task The Same Way

1. Resolve the work item:
   - Non-trivial work starts from a GitHub issue with testable acceptance
     criteria. Issues are generated from
     `docs/plans/2026-05-22-auto-research-implementation.md`.
   - Tiny editorial fixes (typos, broken links, comment polish) skip the issue.
2. If the change spans more than one milestone day, create or enter a worktree:
   `~/Documents/projects/auto-research/.worktree/N-short-slug/` on branch
   `feat/N-short-slug` from `origin/main`. W1 (sequential foundation work) can
   stay in the main checkout. The `.worktree/` directory is git-ignored
   (added in Issue 1's `.gitignore`).
3. **Plan the issue just-in-time** (see §1.5 below). Do not pre-write
   implementation code in the project plan.
4. Classify risk using the 2 tiers below. When uncertain, choose Tier 2.
5. Pick the artifact for that tier (most issues need none beyond the issue
   body + a failing test).
6. Implement in the worktree, **tests first** for behavior changes.
7. Verify using the tier gate and the issue acceptance criteria.
8. Open or update the PR with evidence that maps to the AC.

## 1.5 Plan lifecycle

This repo distinguishes two plan artifacts. Conflating them is the failure
mode that prompted the v1-detailed plan to be archived.

**Project plan** — `docs/plans/2026-05-22-auto-research-implementation.md`.
Issue-shaped: title, objective, acceptance criteria, labels, milestone,
blocked-by. **No implementation code.** Stable; updated only when issues are
added, removed, or reordered. Source of GitHub issues. ~500-700 lines.

**Per-issue plan** — generated **at issue pickup, inside the worktree** by
invoking `superpowers:writing-plans` with the issue body as input. Contains
the bite-sized TDD steps, file paths, test names, commit messages for that
one issue, derived against the actual current state of the codebase.
Disposable. Lives in the PR body or under `docs/plans/per-issue/<N>-<slug>.md`
in the worktree, and is **deleted at PR merge**. Only the AC, commits, and
PR body survive.

**Authority order between them:** AGENTS.md §2 > contract docs > project plan
> per-issue plan. A per-issue plan that contradicts the project plan's AC
loses; an AC that contradicts an invariant loses.

**When to invoke `writing-plans`:** at issue pickup, never at session start.
Pre-writing a plan against a non-existent codebase produces speculative code
that anchors the implementation away from what reading the actual repo would
suggest. The archived
`docs/plans/archive/2026-05-22-auto-research-implementation-v1-detailed.md`
demonstrates the failure mode: 5,500 lines of code written before the first
line of `src/auto_research/` existed.

**When NOT to invoke `writing-plans`:** Tier 0 (editorial), Tier 1 issues
with ≤ 5 trivial steps where the AC alone is enough. The skill is for issues
where the implementation path is genuinely non-obvious.

---

## 2. Risk Tiers

Two tiers. No Tier 3/4 — no live capital, no shared infrastructure.

| Tier | Scope | Artifact | Verification |
|---|---|---|---|
| **0 — Editorial** | Typos, link fixes, comments, README polish, formatting. No behavior change. | Direct patch. Issue optional. | Diff inspection. |
| **1 — Ordinary code** | New code or edits that don't touch a sensitive path (`AGENTS.md` §3). Includes most signal-tweak, ingest, observability, MCP-tool, and agent-graph work. | Issue with AC. PR body lists the AC and the test names that prove each one. | `make quick` (ruff + mypy) + targeted unit test, all green before PR. |
| **2 — Sensitive paths** | Any change under the §3 list: extract guardrails/schemas, Feast feature views, backtest CPCV/deflated-Sharpe/labels/costs, research-graph T1/T2 gates, reliability primitives, MCP server tool surface. | Issue with AC + a one-paragraph Change Contract in the PR body (problem, scope, invariants touched, verification, rollback). Spec only if the design isn't obvious from the issue. | Failing test or eval delta written **first**, then code. PR body cites the test name and shows green pytest output (or Ragas/DeepEval score delta). |

### Path → tier matrix

| Path | Default tier | Escalators |
|---|---|---|
| Markdown, comments | 0 | → 1 if it changes policy or invariants |
| `docs/specs/`, `docs/plans/`, `docs/decisions/`, `AGENTS.md`, `CLAUDE.md`, `docs/AI_WORKFLOW.md`, `docs/AI_CODE_STYLE.md` | 1 | → 2 if it changes an invariant |
| `src/auto_research/ingest/`, `src/auto_research/eval/`, observability glue, `dashboard.py` | 1 | — |
| `src/auto_research/extract/` (worker bodies) | 1 | → 2 for `guardrails.py`, `schemas.py`, citation-grounding logic |
| `feast/` (FeatureView definitions) | 2 | — PIT discipline |
| `src/auto_research/backtest/cpcv.py`, `deflated_sharpe.py`, `labels.py`, `costs.py` | 2 | — López de Prado correctness |
| `src/auto_research/agents/research_graph.py` | 2 | — `T1_GATE`/`T2_GATE` are code-checked authority |
| `src/auto_research/agents/reliability.py` | 2 | — cost cap, circuit breaker, fallback |
| `src/auto_research/mcp_server.py` | 2 | — read-only tool surface; never add write tools |
| `signals/` (A1, A2, B1, combiner) | 1 | → 2 if changing the IC-weighted combiner math |

### Tier escalators

Escalate to Tier 2 when a change:

- Affects PIT correctness, lookahead protection, or `as_of_ts` semantics.
- Touches citation-grounding (`source_span` / `source_quote`).
- Changes promote/iterate/kill thresholds, CV folding, or Sharpe deflation.
- Adds an LLM call to a promote/iterate/kill decision (don't — see INV-3).
- Adds a write-capable MCP tool (don't — the surface is read-only).
- Modifies cost-model plumbing into vbt.

---

## 3. Artifacts

### Change Contract (Tier 2 only)

In the PR body (no separate file needed):

```markdown
## Change Contract
- Tier: 2
- Problem:
- Scope:
- Invariants touched: (e.g. INV-2 citation grounding)
- Verification: (test name + green output, eval-score delta)
- Rollback: (one line — usually `git revert`)
```

### Spec

Use only when the design shape isn't obvious from the issue and a plan-mode
brainstorm would otherwise repeat itself. Specs live in `docs/specs/`. The
canonical project spec is already written; new specs are rare and additive.

### Implementation plan

The 32-task plan is in `docs/plans/`. New plans only for work that spans
multiple issues with ordering constraints.

### ADR

Use for framework-judgment decisions worth pointing an interviewer at:
LangGraph vs Pydantic AI, vbt.pro vs custom, Feast vs hand-rolled,
contextual-chunking vs naive. ADRs live in `docs/decisions/` and follow the
standard format (Context / Decision / Consequences). They're light — half a
page, not five.

### PR

Every PR must contain:

- Summary of what changed and why.
- `Closes #N` line.
- AC mapping — one bullet per AC, each citing a file:line or test name.
- For Tier 2: the Change Contract block.
- Doc-sync note: "no docs affected" or "updated `docs/X.md` to match."

---

## 4. Subagent Policy

Two profiles only. Solo project, four-week budget.

| Profile | Use | Permissions |
|---|---|---|
| `explorer` | Narrow read-only research — caller maps, doc consistency, "where does X live", external-library API lookup. | Read-only. |
| `reviewer` | Independent critique of a Tier 2 PR diff or a draft spec, before merge. | Read-only; no GitHub state changes. |

Code-writing subagents are off by default. The cost of integrating their output
exceeds the benefit at this scope. If a task genuinely splits into independent
write streams (e.g., three signal pytest fixtures), the user will say so.

---

## 5. Verification

| Tier | Gate |
|---|---|
| 0 | Diff inspection. |
| 1 | `make quick` + targeted unit test. |
| 2 | Failing test first, full pytest suite for the touched module, plus the relevant eval gate (DeepEval for extraction, Ragas for RAG, `T1_GATE`/`T2_GATE` for signals/backtest). PR body cites the test name. |

`make quick` = `ruff check .` + `mypy src/auto_research/`.
`make check` = `make quick` + `pytest tests/` + relevant eval suite.

### Sensitive-path evidence template

For Tier 2 PRs, the PR body names each applicable evidence row. State the
non-applicability reason in one line if a row doesn't apply.

| Evidence row | Required for | Form |
|---|---|---|
| Unit test name + green output | All Tier 2 | `tests/path/test_foo.py::test_bar` |
| Property test or stateful test | `cpcv.py`, `deflated_sharpe.py`, `labels.py` | Hypothesis test name + `max_examples` |
| DeepEval score delta | `extract/guardrails.py`, `schemas.py` | Baseline vs new (F1, hallucination rate) |
| Ragas score delta | `extract/chunking.py`, `extract/rag_retrieval.py` | `context_recall`, `faithfulness` |
| Backtest tier gate output | `signals/`, `backtest/engine.py` | `T1_GATE` / `T2_GATE` pass per signal |

---

## 6. Operational Prohibitions

See `AGENTS.md` §5. Repeated here in one line: no secret reads, no destructive
ops on `data/`, no backfill without confirmation, no force-push, no PR merges,
no LLM in trading-decision path.

---

## 7. Third-Party Skills

Superpowers skills are actively used:

- `brainstorming` — start of a new spec or open-ended design question.
- `writing-plans` — when a multi-day milestone needs decomposition.
- `executing-plans` — when working through a plan with checkpoints.
- `subagent-driven-development` — when issues genuinely parallelise.
- `verification-before-completion` — before claiming a Tier 2 PR is ready.
- `test-driven-development` — for any code change touching a sensitive path.

When a skill's required artifact would be heavier than this workflow requires
(e.g., a spec for a typo, a plan for a 5-line fix), follow this workflow and
move on. The repo policy is the floor; skills can add structure above it.

---

## 8. Done Definition

An AI-assisted task is done when:

- Issue acceptance criteria are satisfied with file:line or test evidence.
- Code changes have tests first and pass the relevant gate.
- Tier 2 PRs include the Change Contract and named evidence.
- Docs changes don't duplicate canonical policy (link, don't restate).
- The PR body is current with the branch head.
