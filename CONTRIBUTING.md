# Contributing

This is a personal project, but the workflow is public-facing — both
because the repo is public from day 1 and because the structure itself
is part of the interview-portfolio narrative. Read [`AGENTS.md`](AGENTS.md)
and [`docs/AI_WORKFLOW.md`](docs/AI_WORKFLOW.md) before non-trivial work.

## Issue workflow

Planned milestone work flows through GitHub issues bulk-created from
[`docs/plans/2026-05-22-auto-research-implementation.md`](docs/plans/2026-05-22-auto-research-implementation.md)
via `scripts/create_all_issues.py`. Each issue carries an objective,
testable acceptance criteria, labels, milestone, and blocked-by
references that auto-track closure.

Ad-hoc issues (bugs, follow-ups, polish) use the issue template at
[`.github/ISSUE_TEMPLATE/issue.md`](.github/ISSUE_TEMPLATE/issue.md).

## Worktree convention

Main checkout stays at `~/Documents/projects/auto-research/` on `main`.

Per-issue worktrees live under `.worktree/` (git-ignored):

```bash
# Start an issue (or use the `worktree` skill in Claude Code)
git fetch origin main
git worktree add .worktree/<N>-<slug> -b feat/<N>-<slug> origin/main
cd .worktree/<N>-<slug>
uv sync --all-extras
```

On PR merge:

```bash
git worktree remove .worktree/<N>-<slug>
git branch -d feat/<N>-<slug>
```

W1 D1 sequential foundation work (e.g., this Issue 1) and editorial
fixes are exempt — work directly on `main`.

## Branch + commit format

- Branch: `feat/<N>-<short-slug>` (or `fix/`, `chore/`, `docs/`).
- Commits: conventional commits — e.g., `feat(extract): ten-k worker with citation grounding`.
- PRs: `Closes #N`. Body maps each AC to file:line or test-name evidence.
  Tier 2 PRs include the Change Contract block (see
  [`.github/pull_request_template.md`](.github/pull_request_template.md)).

## Per-issue planning

The project plan stays at AC level. Bite-sized implementation steps are
generated **at issue pickup, inside the worktree** via
`superpowers:writing-plans`. These per-issue plans are disposable —
they live in the PR body or under `docs/plans/per-issue/<N>-<slug>.md`
and are deleted at PR merge. Only the AC, commits, and PR body survive.

See [`docs/AI_WORKFLOW.md`](docs/AI_WORKFLOW.md) §1.5 for the full plan
lifecycle.

## Verification

- `make quick` — ruff + mypy. Run constantly during dev.
- `make check` — quick + pytest (unit, excludes `eval` and `integration` markers).
- `make eval` — paid-API + integration suites. Run locally; excluded from CI.

Tier 2 PRs (sensitive paths per
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §3) require a failing
test first and explicit evidence in the PR body — see
[`docs/AI_WORKFLOW.md`](docs/AI_WORKFLOW.md) §5.
