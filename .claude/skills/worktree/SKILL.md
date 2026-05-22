---
name: worktree
description: Use when starting work on a GitHub issue (creates a per-issue git worktree at the project's canonical path with the right branch name) or cleaning up after a PR merges. Maps to spec §23.4. Skip for W1 D1 sequential foundation work or pure docs edits.
argument-hint: "<start|done|check> <issue-number> [short-slug]"
allowed-tools: Bash
---

# worktree

Manage per-issue git worktrees so parallel Claude Code / Cursor sessions can
work on independent issues without stepping on each other. Particularly useful
in W2-W3 of the plan where RAG, signals, and backtest work advance
concurrently.

## Naming convention (spec §23.4)

- Main checkout: `~/Documents/projects/auto-research/` — stays on `main`.
- Per-issue worktree:
  `~/Documents/projects/auto-research/.worktree/<N>-<short-slug>/`
- Branch name: `feat/<N>-<short-slug>` (or `fix/`, `docs/`, `chore/`
  per conventional-commits prefix).
- Issue number `N` is the GitHub issue number. Short slug is 2-4 words,
  kebab-case, matches the issue title.

## When to skip

- **W1 D1 work** (repo init, pyproject, Docker compose, MLflow wiring) — pure
  sequential foundation, no parallelism gain.
- **Editorial fixes** (typos, README polish) — Tier 0, work on `main` directly.
- **Hot-fix on `main`** when CI is red on production cron — branch from
  `main` in place; document the deviation in the PR.

## `start` — create a worktree for an issue

Required args: `<issue-number> <short-slug>`. The slug is verified against
the issue title.

```bash
# Verify the issue exists and pull its title
gh issue view <N> --json title,labels,milestone

# Create the worktree from origin/main (always from current main, never from a stale local main)
git fetch origin main
git worktree add \
  ~/Documents/projects/auto-research/.worktree/<N>-<slug> \
  -b feat/<N>-<slug> origin/main

# Hand off — print the cd command for the user
echo "Worktree ready. Open a new Claude Code / Cursor session with:"
echo "  cd ~/Documents/projects/auto-research/.worktree/<N>-<slug>"
```

After creation, the new session should:

1. `git branch --show-current` — confirm `feat/<N>-<slug>`.
2. `pwd` — confirm path ends in `<N>-<slug>`.
3. Read the issue body (`gh issue view <N>`) for acceptance criteria.

## `done` — clean up after PR merge

Required args: `<issue-number>`. Slug is recovered from the worktree path.

```bash
# Resolve the worktree path
WORKTREE=$(git worktree list --porcelain | awk -v n="<N>" '
  /^worktree/ { path=$2 }
  /^branch/ && $0 ~ ("feat/" n "-") { print path; exit }
')

if [ -z "$WORKTREE" ]; then
  echo "No worktree found for issue <N>. Already cleaned up?"
  exit 0
fi

# Verify the PR for this issue is merged before deletion
PR_STATE=$(gh pr list --search "<N> in:title" --state merged --json number --jq '.[0].number')
if [ -z "$PR_STATE" ]; then
  echo "PR for issue <N> not merged yet. Refusing to delete worktree."
  echo "If you need to abandon the worktree, run: git worktree remove $WORKTREE --force"
  exit 1
fi

# Remove the worktree (preserves the branch ref; safe to re-checkout if needed)
git worktree remove "$WORKTREE"

# Update local main to the merged state
git -C ~/Documents/projects/auto-research fetch origin main
git -C ~/Documents/projects/auto-research checkout main
git -C ~/Documents/projects/auto-research pull --ff-only origin main

# Optional: delete the local branch ref now that it's merged
git -C ~/Documents/projects/auto-research branch -d feat/<N>-<slug> 2>/dev/null || true

echo "Cleaned up worktree for issue <N>."
```

## `check` — am I in the right worktree?

Run at the start of any new session that claims to be working on a specific
issue. Catches the "agent dispatched into main checkout instead of the
worktree" failure mode.

```bash
PWD_PATH=$(pwd)
BRANCH=$(git branch --show-current)

# Expected: PWD matches ~/Documents/projects/auto-research/.worktree/<N>-<slug>
# Expected: BRANCH matches feat/<N>-<slug>
if [[ "$PWD_PATH" =~ /auto-research/\.worktree/([0-9]+)-([a-z0-9-]+)$ ]]; then
  ISSUE_N="${BASH_REMATCH[1]}"
  SLUG="${BASH_REMATCH[2]}"
  EXPECTED_BRANCH="feat/${ISSUE_N}-${SLUG}"
  if [ "$BRANCH" != "$EXPECTED_BRANCH" ]; then
    echo "MISMATCH: path says issue #${ISSUE_N} (${SLUG}), branch is ${BRANCH}"
    exit 1
  fi
  echo "OK: issue #${ISSUE_N}, branch ${BRANCH}"
elif [[ "$PWD_PATH" =~ /auto-research$ ]] && [ "$BRANCH" = "main" ]; then
  echo "Main checkout on main. OK for editorial / W1 D1 / hot-fix work only."
else
  echo "UNKNOWN STATE: pwd=${PWD_PATH} branch=${BRANCH}"
  echo "If you're an agent dispatched for issue work, the parent likely sent you to the wrong directory."
  exit 1
fi
```

## Pre-commit guard

Any commit on a `feat/<N>-` branch should reference the issue in either the
commit body or the PR body. The agent should verify before pushing:

```bash
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" =~ ^feat/([0-9]+)- ]]; then
  ISSUE_N="${BASH_REMATCH[1]}"
  # Either the commit message or a later PR body must reference #<N>
  echo "Reminder: commit or PR body should reference #${ISSUE_N}"
fi
```

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `git worktree add` fails: "already checked out" | Worktree from a prior session not cleaned up | `git worktree list`, then `git worktree remove <path>` or `--force` if dirty |
| Branch creation fails: "already exists" | Earlier session created the branch but the worktree was removed | `git worktree add <path> feat/<N>-<slug>` (no `-b`) to reuse |
| New session lands in `~/Documents/projects/auto-research/` (main) but branch is `feat/...` | User opened a terminal in the wrong dir | `cd ~/Documents/projects/auto-research/.worktree/<N>-<slug>` |
| Agent edits files in main checkout while a worktree exists for the same issue | Skill not invoked at session start | Run `worktree check` before any non-trivial edit |
