<!--
Delete sections marked [conditional] when they don't apply.
Authority: AGENTS.md §2 invariants > docs/AI_WORKFLOW.md > this template.
-->

## Summary

<!-- 1-3 bullets: what changed and why. -->

-

Closes #

## Tier

<!-- Per docs/AI_WORKFLOW.md §2. Tier 2 requires the Change Contract below. -->

- [ ] Tier 0 — editorial
- [ ] Tier 1 — ordinary code
- [ ] Tier 2 — sensitive path touched (one of: extract/guardrails, extract/schemas, feast_repo/, backtest/{cpcv,deflated_sharpe,labels,costs,gates}, agents/{research_graph,reliability}, mcp_server)

## Acceptance criteria evidence

<!-- One row per AC from the issue. Evidence = test name, file:line, or eval delta. -->

| AC | Evidence |
|---|---|
| 1. | `tests/path/test_x.py::test_y` (passed) |
| 2. |  |

## Verification

<!-- Check what actually ran. Don't check what you didn't run. -->

- [ ] `make quick` (ruff + mypy) passes
- [ ] `make check` passes (unit tests; eval suite excluded by default)
- [ ] Relevant skill invoked: `pit-check` / `citation-check` / `bump-prompt-version` / `worktree`
- [ ] `gh pr checks` is green

## Change Contract  [conditional — Tier 2 only]

<!-- Required for sensitive-path PRs. Delete if Tier 0 or 1. -->

- **Problem:**
- **Scope:**
- **Invariants touched:** <!-- e.g. INV-1 PIT, INV-2 citation, INV-6 prompt version -->
- **Verification:** <!-- named test + green output -->
- **Rollback:** <!-- usually `git revert <SHA>`; note anything else (cache invalidation, re-materialization) -->

## Sensitive-path evidence  [conditional]

<!-- Required if any path marked [SENSITIVE] in docs/ARCHITECTURE.md §3 was edited. -->

| Layer | Evidence |
|---|---|
| Unit / property test | `tests/...::test_...` |
| DeepEval delta *(extract)* | `hallucination_rate: 0.04 → 0.02` |
| Ragas delta *(rag)* | `context_recall: 0.78 → 0.83` |
| Backtest gate pass *(signals)* | `T2_GATE` row results vs threshold |

## Eval delta  [conditional]

<!-- Required if extract/, agents/, signals/, or backtest/ behavior changed. -->

| Metric | Baseline | New | Delta |
|---|---|---|---|
|  |  |  |  |

## Doc-sync

<!-- One line. Mark "no docs affected" or name the updated docs. -->

-

## Skipped checks

<!-- Default: None. If you skipped a check, name the exact reason. -->

- None

---

<!--
Post-merge:
- `worktree done <N>` cleans up the per-issue worktree.
- For Tier 2 PRs on sensitive paths, confirm no `prompt_version` bump was forgotten.
- For backfill-touching PRs, re-materialize Feast: `feast materialize-incremental ...`
-->
