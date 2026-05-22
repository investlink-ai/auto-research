# Session handoff — auto-research, 2026-05-22

> **Purpose:** Capture the in-flight state of the auto-research project so a fresh Claude Code session can pick up cleanly. Brainstorming + spec are done; writing-plans was interrupted by a model classifier outage with Milestones 3-4 of the implementation plan still unwritten.

## Where we are

**Workflow stage:** brainstorming → spec → **writing-plans (in progress, blocked at Milestone 3)**.

The plan file at `docs/plans/2026-05-22-auto-research-implementation.md` contains the full plan header and **Milestones 1-2 (Tasks 1-22)** with bite-sized TDD steps, exact file paths, full code blocks, and commit messages.

**What's missing:** Milestones 3-4 (Tasks 23-32). All design decisions for those tasks are locked in the spec; the work to do is to re-derive the bite-sized steps and append them to the plan file.

## What's persisted on disk

- `~/Documents/projects/auto-research/` — standalone git repo (`main` branch).
- `docs/specs/2026-05-22-design.md` — 24-section approved design spec.
- `docs/plans/2026-05-22-auto-research-implementation.md` — implementation plan, Milestones 1-2 only.
- Commits: `95a7200` (initial spec), `7abca73` (PM section added to spec), plus the initial plan commit.
- No source code yet — implementation has not begun.

## Locked decisions (so the next session doesn't re-litigate)

| Decision | Value |
|---|---|
| Repo name | `auto-research` (standalone, public on GitHub) |
| Build time | 4 weeks full-time, ~160 hrs total, heavy Claude Code use |
| Universe | ~90 names, two sub-universes — AI infrastructure supply chain (~70) + frontier tech speculative (~20) |
| Backfill window | 2 years (Jan 2024 – present) |
| Architecture | Approach 1: stateless ETL extraction + LangGraph only on research/critic loops (PDF two-plane) |
| Feature store | Feast 0.43+ on Parquet, offline-only |
| Backtest engine | vectorbt.pro + custom CPCV/deflated-Sharpe/cost layers on top |
| Signal families | A1 cross-doc supply-chain forward-tone, A2 PEAD language drift, B1 frontier milestone/dilution |
| Validation tiers | T1 info tests → T2 portfolio backtest → T3 stress |
| Decide gates | Code-checked (`T1_GATE`, `T2_GATE` constants); LLM critic produces only qualitative addendum |
| Research agent | LangGraph with checkpointer + HITL interrupt; 6 hypothesis types |
| Live critic | Pydantic AI (not LangGraph) — framework-judgment story |
| LLM ops | Anthropic API tiered (Haiku/Sonnet/Opus) + Batch + prompt caching; ~$100-200 total |
| MCP server | FastMCP, exposes query_features / run_backtest / search_memos / list_alpha_library / read_signal_performance |
| RAG | unstructured.io + Anthropic contextual chunking + LanceDB + Voyage embeddings + BM25/dense/RRF + BGE reranker + Ragas |
| Eval | DeepEval (pytest-style) for extraction, Ragas for RAG |
| Guardrails | Guardrails AI + custom citation grounding (source_span + source_quote on every claim) |
| Observability | Langfuse self-hosted (Docker) + OpenTelemetry/OpenInference + MLflow alpha library |
| Optional stretch | Streamlit dashboard (Task 32, cut candidate) |
| Transcripts source | FMP paid API |
| Project mgmt | GitHub public repo, 4 milestones (one per week), ~32 issues, worktree-per-issue workflow |

## What Milestones 3-4 still need

Append to the plan file in `## Milestone 3 — W3: Signals + backtest gauntlet` and `## Milestone 4 — W4: Research agent + live critic + MCP + polish` sections.

Tasks to flesh out (each follows the same bite-sized TDD pattern as Tasks 1-22):

**Milestone 3 (W3):**
- Task 23 — T1 info_tests primitives (event_study, ic_analysis, quantile_sort, conditional_distribution, mutual_information, bootstrap_significance) in `src/auto_research/backtest/info_tests.py`.
- Task 24 — Triple-barrier labels + CPCV with embargo + deflated Sharpe + cost model + `InfoReport` / `BacktestReport` dataclasses.
- Task 25 — Minimal backtest engine wrapping vbt.pro pattern + tier-aware `check_t1_gate` / `check_t2_gate` with code-checked thresholds (no LLM judgment).
- Task 26 — Signals A2 (PEAD language drift), A1 (supply-chain forward-tone with time decay), B1 (frontier milestone/dilution) + IC-weighted combiner with Ledoit-Wolf shrinkage + `AlphaLibrary` backed by MLflow runs.

**Milestone 4 (W4):**
- Task 27 — FastMCP server `src/auto_research/mcp_server.py` exposing the read-only data + research interface.
- Task 28 — LangGraph research agent: `ResearchState`, `HypothesisType` enum, state-machine nodes (propose_hypothesis → materialize_signal → run_validation → decide || critique → write_memo), SqliteSaver checkpointer wiring.
- Task 29 — Pydantic AI live critic with `LiveCriticInput` / `LiveCriticOutput` schemas, TestModel for unit tests; two worked-example scripts (one promoted, one killed) that produce attribution memos.
- Task 30 — Rewrite README (PDF-v0-critique framing), `docs/architecture.md` (port two-plane diagram), `docs/signal_cards/{A2,A1,B1}_*.md` (one card per signal).
- Task 31 — `scripts/setup_github.sh` for repo + milestones + labels; `scripts/create_all_issues.py` to bulk-create issues from this plan file.
- Task 32 — Optional Streamlit dashboard (`dashboard.py`).

Self-review checklist at the end (spec coverage, placeholder scan, type consistency).

## Prompt to start the new session

Copy-paste the following into the fresh Claude Code session at `~/Documents/projects/interview_prep/` (or anywhere — the prompt references absolute paths):

> Resume work on the auto-research project. Read `~/Documents/projects/auto-research/docs/SESSION_HANDOFF.md` for the current state, then read `~/Documents/projects/auto-research/docs/specs/2026-05-22-design.md` for the locked design and `~/Documents/projects/auto-research/docs/plans/2026-05-22-auto-research-implementation.md` for the plan-so-far (Milestones 1-2 only).
>
> Your task: append Milestones 3 and 4 (Tasks 23-32) to the plan file, following the same bite-sized TDD pattern used in Tasks 1-22 — exact file paths, complete code blocks, test-first ordering, conventional-commit messages. Use the `superpowers:writing-plans` skill for the format. Once appended, do the self-review inline (spec coverage check, placeholder scan, type consistency), commit, then tell me the plan is ready for review before we move to implementation.
>
> The previous session ran into a model classifier outage mid-write; the design + first two milestones are stable and committed. No need to re-brainstorm. Don't re-init the repo; it exists at `~/Documents/projects/auto-research/` already.

## Memory the next session should pick up automatically

These auto-memory entries are already saved and will load on session start:

- `feedback_cc_speedup_assumption.md` — discount solo-coded hour estimates ~30-40% before quoting
- `feedback_concrete_plans.md` — name tools/URLs and day-by-day actions
- `feedback_framework_preference.md` — LangGraph primary, Pydantic AI secondary
- `feedback_show_uplift_traces.md`, `feedback_no_code_duplication_in_docs.md`, etc.
- `user_profile.md`, `side_project_funding_rate_arb.md`, `english_interview_prep.md`

No new memories needed for the resume.

## Notes for the next session

- Don't redo brainstorming. The design is locked; only Milestones 3-4 of the plan remain.
- The `claude-opus-4-7` classifier outage that blocked this session was provider-side; should be cleared by the time you resume.
- After the plan is complete + reviewed + approved, the terminal state of writing-plans is to offer subagent-driven-development vs inline-execution for implementation. That's the next decision after plan approval.
- Implementation has not started. No code in `src/auto_research/` yet.
