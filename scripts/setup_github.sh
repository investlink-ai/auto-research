#!/usr/bin/env bash
# Idempotent GitHub bootstrap: create repo + 4 milestones + 15 labels.
#
# Prereqs:
#   - gh CLI authenticated under the right account (e.g. `gh auth switch -u feynman0825`)
#   - Run from repo root, where the local git repo already exists
#
# Usage:
#   scripts/setup_github.sh
#
# Env overrides:
#   REPO_NAME      (default: auto-research)
#   GITHUB_OWNER   (default: $(gh api user --jq .login))

set -euo pipefail

REPO_NAME="${REPO_NAME:-auto-research}"
OWNER="${GITHUB_OWNER:-$(gh api user --jq .login)}"
DESCRIPTION="Two-plane multi-agent research platform for cross-asset language-driven alpha in AI infrastructure and frontier-tech equities"

echo "Target: $OWNER/$REPO_NAME"
echo

# 1. Repo --------------------------------------------------------------------
if gh repo view "$OWNER/$REPO_NAME" >/dev/null 2>&1; then
  echo "repo exists: $OWNER/$REPO_NAME"
else
  echo "creating repo: $OWNER/$REPO_NAME"
  gh repo create "$OWNER/$REPO_NAME" \
    --public \
    --source=. \
    --remote=origin \
    --push \
    --description "$DESCRIPTION"
fi

# 2. Milestones (one per week from today, 7/14/21/28 days out) ---------------
declare -a MILESTONES=(
  "1:W1 — Foundation + extraction backbone"
  "2:W2 — RAG layer + extraction quality"
  "3:W3 — Signals + backtest gauntlet"
  "4:W4 — Research agent + live critic + MCP + polish"
)

# date is BSD on macOS, GNU on Linux
date_offset() {
  local days="$1"
  if date -u -v+0d +%Y-%m-%d >/dev/null 2>&1; then
    date -u -v+"${days}"d +%Y-%m-%dT23:59:59Z
  else
    date -u -d "+${days} days" +%Y-%m-%dT23:59:59Z
  fi
}

EXISTING_MILESTONES="$(gh api "repos/$OWNER/$REPO_NAME/milestones?state=all" --jq '.[].title' 2>/dev/null || true)"
for entry in "${MILESTONES[@]}"; do
  week="${entry%%:*}"
  title="${entry#*:}"
  due_days=$((7 * week))
  due_date="$(date_offset "$due_days")"
  if echo "$EXISTING_MILESTONES" | grep -qFx "$title"; then
    echo "milestone exists: $title"
  else
    gh api -X POST "repos/$OWNER/$REPO_NAME/milestones" \
      -f title="$title" -f due_on="$due_date" --silent
    echo "created milestone: $title (due $due_date)"
  fi
done

# 3. Labels: domain (color-coded) + size (gray) -----------------------------
# Two parallel arrays so the script works in bash 3.2 (macOS default).
LABEL_NAMES=(infra extract rag signal backtest agent mcp eval obs docs polish extra-small small medium large)
LABEL_COLORS=(cfd3d7 fbca04 0e8a16 1d76db 5319e7 b60205 d4c5f9 fef2c0 bfdadc c2e0c6 ededed ededed ededed ededed ededed)

EXISTING_LABELS="$(gh label list -R "$OWNER/$REPO_NAME" -L 200 --json name --jq '.[].name' 2>/dev/null || true)"
for i in "${!LABEL_NAMES[@]}"; do
  name="${LABEL_NAMES[$i]}"
  color="${LABEL_COLORS[$i]}"
  if echo "$EXISTING_LABELS" | grep -qFx "$name"; then
    echo "label exists: $name"
  else
    gh label create "$name" -c "$color" -R "$OWNER/$REPO_NAME" >/dev/null
    echo "created label: $name"
  fi
done

echo
echo "GitHub bootstrap complete."
echo "Next:"
echo "  uv run python scripts/create_all_issues.py --dry-run   # preview"
echo "  uv run python scripts/create_all_issues.py             # live create"
