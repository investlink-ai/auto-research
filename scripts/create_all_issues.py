"""Parse the implementation plan and create or update GitHub issues.

Idempotent: by default, skips issues whose title already exists. With
`--update`, edits the body of existing issues to match the current plan
content (useful when the plan or this parser has changed).

Run from repo root after `scripts/setup_github.sh`.

Usage:
    python scripts/create_all_issues.py              # create missing
    python scripts/create_all_issues.py --dry-run    # parse + preview only
    python scripts/create_all_issues.py --update     # refresh existing bodies
    python scripts/create_all_issues.py --only 5,7   # specific issues
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs" / "plans" / "2026-05-22-auto-research-implementation.md"
)

MILESTONE_TITLES: dict[str, str] = {
    "W1": "W1 — Foundation + extraction backbone",
    "W2": "W2 — RAG layer + extraction quality",
    "W3": "W3 — Signals + backtest gauntlet",
    "W4": "W4 — Research agent + live critic + MCP + polish",
}

ISSUE_HEADER_RE = re.compile(
    r"^### Issue (?P<n>\d+) — `(?P<title>[^`]+)`",
    re.MULTILINE,
)
OBJECTIVE_RE = re.compile(
    r"\*\*Objective\.\*\*\s*(?P<text>.+?)(?=\*\*Acceptance criteria\.\*\*)",
    re.DOTALL,
)
AC_RE = re.compile(
    r"\*\*Acceptance criteria\.\*\*\s*(?P<text>.+?)(?=\*\*Labels\.\*\*)",
    re.DOTALL,
)
FOOTER_RE = re.compile(
    r"\*\*Labels\.\*\*\s*(?P<labels>.+?)\s*"
    r"\*\*Milestone\.\*\*\s*(?P<milestone>W\d)\s*"
    r"\*\*Blocked by\.\*\*\s*(?P<blocked>.+?)\s*$",
    re.MULTILINE | re.DOTALL,
)
INLINE_BACKTICK_RE = re.compile(r"`([^`]+)`")


@dataclass
class Issue:
    n: int
    title: str
    objective: str
    ac_lines: list[str]
    labels: list[str]
    milestone: str  # W1 / W2 / W3 / W4
    blocked_by: list[int] = field(default_factory=list)


def parse_ac_bullets(ac_text: str) -> list[str]:
    """Parse bullets from an AC block, joining multi-line continuations.

    A bullet starts at a line beginning with "- ". Subsequent non-empty lines
    that do not start with "- " are continuations and get joined with a
    single space. Empty lines flush the current bullet.
    """
    bullets: list[str] = []
    current: str | None = None
    for raw in ac_text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("- "):
            if current is not None:
                bullets.append(current.strip())
            current = stripped[2:].strip()
        elif current is not None and stripped:
            current += " " + stripped
        elif not stripped:
            if current is not None:
                bullets.append(current.strip())
                current = None
    if current is not None:
        bullets.append(current.strip())
    return bullets


def parse_issues(plan_text: str) -> list[Issue]:
    issues: list[Issue] = []
    headers = list(ISSUE_HEADER_RE.finditer(plan_text))
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(plan_text)
        body = plan_text[start:end]

        n = int(h.group("n"))
        title = h.group("title").strip()

        obj_m = OBJECTIVE_RE.search(body)
        objective = obj_m.group("text").strip() if obj_m else ""

        ac_m = AC_RE.search(body)
        ac_text = ac_m.group("text").strip() if ac_m else ""
        ac_lines = parse_ac_bullets(ac_text)

        foot_m = FOOTER_RE.search(body)
        if not foot_m:
            print(
                f"WARN: issue {n} has no Labels/Milestone footer; skipping",
                file=sys.stderr,
            )
            continue
        labels = INLINE_BACKTICK_RE.findall(foot_m.group("labels"))
        milestone = foot_m.group("milestone")
        blocked_text = foot_m.group("blocked")
        blocked_by = (
            [int(m) for m in re.findall(r"#?(\d+)", blocked_text)]
            if blocked_text.strip() != "—"
            else []
        )

        issues.append(
            Issue(
                n=n,
                title=title,
                objective=objective,
                ac_lines=ac_lines,
                labels=labels,
                milestone=milestone,
                blocked_by=blocked_by,
            )
        )
    return issues


def existing_issues_by_title() -> dict[str, int]:
    try:
        out = subprocess.check_output(
            [
                "gh", "issue", "list",
                "--state", "all",
                "--limit", "200",
                "--json", "title,number",
            ],
            text=True,
        )
        return {row["title"]: row["number"] for row in json.loads(out)}
    except subprocess.CalledProcessError as exc:
        print(f"WARN: could not list existing issues: {exc}", file=sys.stderr)
        return {}


def issue_body(issue: Issue) -> str:
    parts = [
        f"Implementation guidance: see [`docs/plans/2026-05-22-auto-research-implementation.md`]"
        f"(./docs/plans/2026-05-22-auto-research-implementation.md) — Issue {issue.n}.\n\n"
        f"Per-issue implementation plan is generated **at pickup** via "
        f"`superpowers:writing-plans` inside the worktree; see "
        f"`docs/AI_WORKFLOW.md` §1.5.\n",
        f"## Objective\n\n{issue.objective}\n",
        "## Acceptance criteria\n\n" + "\n".join(f"- [ ] {ac}" for ac in issue.ac_lines) + "\n",
    ]
    if issue.blocked_by:
        # GitHub task-list syntax: `- [ ] #N` renders as a tracked checkbox
        # that auto-checks when the referenced issue closes.
        parts.append(
            "## Blocked by\n\n"
            + "\n".join(f"- [ ] #{b}" for b in issue.blocked_by)
            + "\n"
        )
    return "\n".join(parts)


def create_or_update(
    issue: Issue,
    *,
    dry_run: bool,
    update: bool,
    existing: dict[str, int],
) -> str:
    title = f"Issue {issue.n}: {issue.title}"
    body = issue_body(issue)
    milestone = MILESTONE_TITLES[issue.milestone]

    if title in existing:
        issue_number = existing[title]
        if not update:
            return f"SKIP (exists): #{issue_number} {title}"
        if dry_run:
            return f"DRY-RUN would update body: #{issue_number} {title}"
        subprocess.check_call(
            ["gh", "issue", "edit", str(issue_number), "--body", body]
        )
        return f"UPDATED: #{issue_number} {title}"

    if dry_run:
        return (
            f"DRY-RUN would create: {title}\n"
            f"  milestone: {milestone}\n"
            f"  labels:    {issue.labels}\n"
            f"  blocked:   {issue.blocked_by or '[]'}"
        )

    cmd = ["gh", "issue", "create",
           "--title", title,
           "--body", body,
           "--milestone", milestone]
    for lbl in issue.labels:
        cmd += ["--label", lbl]
    subprocess.check_call(cmd)
    return f"CREATED: {title}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print without calling gh.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Edit existing issues' bodies in place instead of skipping.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated issue numbers to limit to (e.g. '5,7,12').",
    )
    args = parser.parse_args()

    plan_text = PLAN_PATH.read_text()
    issues = parse_issues(plan_text)
    print(f"Parsed {len(issues)} issues from {PLAN_PATH.name}")

    if not issues:
        print("ERROR: no issues parsed; check plan format", file=sys.stderr)
        return 1

    if args.only:
        wanted = {int(x) for x in args.only.split(",") if x.strip()}
        issues = [i for i in issues if i.n in wanted]
        print(f"Filtered to {len(issues)} issues: {sorted(i.n for i in issues)}")

    existing = existing_issues_by_title() if not args.dry_run else {}

    counts = {"CREATED": 0, "UPDATED": 0, "SKIP": 0, "DRY-RUN": 0}
    for issue in issues:
        result = create_or_update(
            issue,
            dry_run=args.dry_run,
            update=args.update,
            existing=existing,
        )
        print(result)
        for key in counts:
            if result.startswith(key):
                counts[key] += 1
                break

    print(
        f"\nSummary: "
        f"{counts['CREATED']} created, "
        f"{counts['UPDATED']} updated, "
        f"{counts['SKIP']} skipped, "
        f"{counts['DRY-RUN']} dry-run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
