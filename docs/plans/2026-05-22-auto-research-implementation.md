# auto-research Implementation Plan

| | |
|---|---|
| **Date** | 2026-05-22 |
| **Status** | Issues-ready (objectives + AC only; per-issue plans generated at pickup) |
| **Spec** | `docs/specs/2026-05-22-design.md` (frozen v1 design) |
| **Total issues** | 32 (30 mandatory + 1 bootstrap + 1 stretch) |

## Plan lifecycle (read first)

This plan is **issue-shaped**. Each section below is one GitHub issue —
title, objective, acceptance criteria, labels, milestone, blocked-by. **No
implementation code.** Implementation code is a just-in-time artifact
produced at issue pickup, not at plan time.

When you pick up an issue:

1. `worktree start <N> <slug>` (or work in the main checkout for W1 D1).
2. Re-read the relevant `AGENTS.md` invariants and the relevant contract
   doc (`docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, `docs/DATA_MODEL.md`,
   `docs/BACKTEST.md`).
3. Run `superpowers:writing-plans` *for that one issue*, with the issue
   body as input. The skill produces bite-sized TDD steps **in the
   worktree, against the current codebase**.
4. The per-issue plan lives in the PR body or under
   `docs/plans/per-issue/<N>-<slug>.md` and is disposable — deleted at PR
   merge. Only the AC, the commits, and the PR body survive.
5. Run `superpowers:executing-plans` (or `subagent-driven-development` for
   parallel branches) to implement.

The reason this plan is issue-shaped and not implementation-shaped:
the agent that drafted the original detailed plan worked from the spec
alone, before any code existed. Library API guesses, test fixture shapes,
and helper-function names from that draft were necessarily speculative.
The detailed v1 is preserved at
`docs/plans/archive/2026-05-22-auto-research-implementation-v1-detailed.md`
as a learning artifact — it shows the failure mode of pre-writing.

Authority on conflict: AGENTS.md §2 > contract docs > this plan > per-issue
plans.

## Bootstrap (run once, before any issue work)

The script that creates the GitHub repo, milestones, labels, and the 31
issues below is itself Issue 31. To break the chicken-and-egg:

1. Create Issue 31 by hand on GitHub (after the repo exists).
2. Implement Issue 31 (write the two scripts + run them).
3. Closing Issue 31's PR populates Issues 1-30 and 32.

After bootstrap, the plan-to-issue mapping is 1:1.

## Conventions

- **Issue title** = conventional-commits form (e.g.,
  `feat(extract): pydantic schemas with citation grounding`).
- **Branch** = `feat/<N>-<slug>` from `origin/main` per `worktree` skill.
- **PR** = `Closes #N`; body maps to AC; Tier 2 PRs include the Change
  Contract block per `docs/AI_WORKFLOW.md` §3.
- **Sensitive paths** (marked `[SENSITIVE]` below) are Tier 2 per
  `docs/AI_WORKFLOW.md` §2 — failing test first, named evidence in PR body.
- **Labels:** `infra`, `extract`, `rag`, `signal`, `backtest`, `agent`,
  `mcp`, `eval`, `obs`, `docs`, `polish` + size `xs`/`s`/`m`/`l`.

---

## Milestone 1 — W1: Foundation + extraction backbone

### Issue 1 — `chore(repo): scaffold uv project with pyproject, ruff, mypy, pytest`

**Objective.** Bootstrap the Python project: `uv` venv, `pyproject.toml`,
ruff, mypy, pytest, `Makefile`, `.gitignore`, `.env.example`, README stub,
`CONTRIBUTING.md` documenting the worktree convention. After this lands,
`.github/workflows/ci.yml` (already committed) starts running.

**Acceptance criteria.**
- `uv sync --all-extras` succeeds from a clean clone.
- `Makefile` defines `quick` (ruff check + mypy), `check` (= quick + pytest
  excluding `eval` and `integration` markers), and `eval` (run eval suite).
- `pyproject.toml` declares pytest markers under
  `[tool.pytest.ini_options]`: `eval` (paid-API tests, excluded from CI)
  and `integration` (external-service tests, excluded from CI).
- `make quick` returns clean on the empty `src/auto_research/`.
- `make check` passes with an empty suite.
- `uv run python -c "import auto_research; print(auto_research.__version__)"` prints `0.1.0`.
- `.env.example` lists every required external service key (Anthropic, FMP,
  Voyage, Langfuse, SEC user-agent) with empty values; no real secrets committed.
- `CONTRIBUTING.md` documents the worktree convention from spec §23.4.
- This PR's CI run is green (proves the workflow + Makefile + markers wire up).

**Labels.** `infra`, `s` **Milestone.** W1 **Blocked by.** —

### Issue 2 — `chore(obs): Langfuse self-hosted via Docker compose + OpenLLMetry wiring`

**Objective.** Stand up Langfuse self-hosted via Docker compose; wire
OpenLLMetry / OpenInference instrumentation so Anthropic + LangChain calls
auto-trace to Langfuse via OTLP.

**Acceptance criteria.**
- `docker compose up -d` brings Langfuse up at `http://localhost:3000`.
- `src/auto_research/telemetry.py` exposes `init_telemetry()` that wires
  OpenLLMetry once per process (idempotent).
- A smoke test calls Anthropic via the SDK and the trace appears in Langfuse
  with token counts populated.
- README quickstart documents `docker compose up` as W1 setup step.

**Labels.** `infra`, `obs`, `s` **Milestone.** W1 **Blocked by.** #1

### Issue 3 — `chore(obs): MLflow local file backend + smoke test`

**Objective.** Initialize MLflow file backend at `mlruns/`, expose a thin
`src/auto_research/experiment.py` wrapper used by `backtest/` and
`agents/alpha_library.py`.

**Acceptance criteria.**
- `MLFLOW_TRACKING_URI=file:./mlruns` set in `.env.example`.
- `auto_research.experiment.start_run(...)` context manager logs params + a
  dummy artifact; the run is visible via `mlflow ui`.
- Test asserts a logged param round-trips.

**Labels.** `infra`, `obs`, `s` **Milestone.** W1 **Blocked by.** #1

### Issue 4 — `feat(data): universe loader + ticker registry`

**Objective.** Implement `auto_research.universe.load_universe()` reading
`data/universe/universe_v1.json` (spec §5: ~70 AI infra + ~20 frontier tech).
Universe entries carry sub-universe, sector, market-cap tier, and an
explicit `tradeable: bool` flag (narrative-source names default to `False`).

**Acceptance criteria.**
- `data/universe/universe_v1.json` checked in with the ~90-name list from spec §5.
- `load_universe()` returns frozen Pydantic models; mutation raises.
- `load_universe(tradeable_only=True)` filters narrative-source names out.
- Tests cover empty universe (raises), duplicate ticker (raises), unknown
  sub-universe (raises).

**Labels.** `infra`, `s` **Milestone.** W1 **Blocked by.** #1

### Issue 5 — `feat(ingest): EDGAR client for 10-K/Q, 8-K, S-1/S-3 + manifest ledger`

**Objective.** `auto_research.ingest.edgar` fetches 10-K, 10-Q, 8-K, S-1,
S-3 from SEC EDGAR public JSON. Records `accepted_datetime` as canonical
PIT stamp. Idempotent on `(cik, accession_number)` via the append-only
manifest at `data/manifest.parquet`.

**Acceptance criteria.**
- Re-running the fetcher with the same `(cik, accession_number)` is a no-op
  (manifest hit).
- Raw bytes persisted at `data/raw/edgar/{cik}/{year}/{accession}.{ext}`.
- Manifest schema includes `content_sha256` for tamper detection.
- VCR-recorded integration test covers one 10-K, one 8-K, one S-3.
- User-Agent header reads from `SEC_USER_AGENT` env (SEC requires it);
  missing env raises a typed error.

**Labels.** `infra`, `s` **Milestone.** W1 **Blocked by.** #1, #4

### Issue 6 — `feat(ingest): FMP transcript client + manifest integration`

**Objective.** `auto_research.ingest.fmp` fetches earnings call transcripts
from FMP. Records `event_datetime` (call start). Coverage gaps on smallest
frontier-tech names recorded as `null` + provenance metadata, not retried
into degraded data.

**Acceptance criteria.**
- Returns frozen `Transcript` model with `event_datetime`, `ticker`,
  `prepared_remarks`, `q_and_a`.
- Missing coverage returns `None` + writes manifest row with `status="no_coverage"`.
- Manifest entries are idempotent (rerun is no-op).
- VCR-recorded test covers one populated transcript and one gap case.

**Labels.** `infra`, `s` **Milestone.** W1 **Blocked by.** #5

### Issue 7 — `feat(feast): Feast scaffold + price FeatureView + PIT discipline property test` [SENSITIVE]

**Objective.** Scaffold `feast_repo/` per `docs/DATA_MODEL.md` §3: entity,
`price_features` FeatureView, `next_trading_day_cutoff` PIT helper, file
offline store. **PIT discipline property test is the gate.**

**Acceptance criteria.**
- `feast apply` succeeds; `price_features` materialized for a synthetic 30-day window.
- `feast_repo/_pit.py::next_trading_day_cutoff` respects NYSE holidays.
- Property test in `tests/feast/test_pit_properties.py` asserts
  `as_of_ts == next_trading_day_cutoff(event_datetime)` for every row,
  generated via Hypothesis over a wide date range.
- No query-time `as_of_ts` arithmetic anywhere in repo (verified by `pit-check` skill).
- PR cites the property test by name.

**Labels.** `infra`, `m` **Milestone.** W1 **Blocked by.** #4

### Issue 8 — `feat(agents): reliability primitives (circuit breaker, cost cap, retry, fallback)` [SENSITIVE]

**Objective.** Build the decorator stack in
`src/auto_research/agents/reliability.py` per `docs/CONTRACTS.md` §5:
`@circuit_breaker`, `@cost_cap`, `@max_iterations`, `@retry_with_backoff`,
`@fallback_model`, and the composite `@reliable_agent_node`.

**Acceptance criteria.**
- Each decorator has a unit test for its trip condition.
- `cost_cap` reads token counts from Anthropic response metadata (real),
  not a mock; uses VCR.
- Composite decorator ordering matches contract (cost_cap outermost).
- Failing `cost_cap` raises `CostCapExceeded`; failing `circuit_breaker`
  raises `CircuitOpen` — both are typed, not generic.

**Labels.** `agent`, `infra`, `m` **Milestone.** W1 **Blocked by.** #1, #2

### Issue 9 — `feat(extract): Pydantic schemas + citation-grounding validator` [SENSITIVE]

**Objective.** Implement `extract/schemas.py` per `docs/CONTRACTS.md` §1
(`Citation`, `Claim`, `TenKOutput`, `TranscriptOutput`, `EightKOutput`,
`SFilingOutput`) and `extract/guardrails.py` with the post-validator that
asserts `source_text[span] == source_quote`.

**Acceptance criteria.**
- All output models frozen; mutation raises.
- `validate_citation_grounding(output, source_text)` raises `CitationMismatch`
  on any failed claim.
- Worker stub demonstrates the quarantine route — failure writes a
  `QuarantineRecord` to `data/quarantine/<worker>/<doc_id>.json` and returns `None`.
- Property test feeds a deliberately corrupted citation and asserts
  quarantine write (per `citation-check` skill).
- No `permissive` / `soft_mode` / `skip_validation` flag exists anywhere.

**Labels.** `extract`, `m` **Milestone.** W1 **Blocked by.** #1

### Issue 10 — `feat(extract): Anthropic client with prompt caching + tiered model routing`

**Objective.** Thin wrapper over the Anthropic SDK that enables prompt
caching (`cache_control` ephemeral on stable prefixes), routes by
worker/task to the model tier from spec §7.3, and integrates with
`agents/reliability.py` decorators.

**Acceptance criteria.**
- VCR test confirms `cache_creation_input_tokens` / `cache_read_input_tokens`
  appear in response metadata after a second call with the same cached prefix.
- `route_model(worker, task)` returns the configured tier; raises on unknown task.
- Reliability decorators applied by default.
- Cost log per call includes `(input_tokens, output_tokens, cache_read, cache_create, est_usd)`.

**Labels.** `extract`, `m` **Milestone.** W1 **Blocked by.** #8, #9

### Issue 11 — `feat(extract): prompts registry directory + S-1/S-3 worker end-to-end`

**Objective.** Establish `extract/prompts/` convention (one file per
prompt, version constant + prompt text) and implement the **S-1/S-3 worker
end-to-end** as the simplest validator of the pipeline (dilution language +
form classification). Wire to Langfuse prompt registry.

**Acceptance criteria.**
- S-1/S-3 worker extracts a real S-3 from `data/raw/` and produces a frozen
  `SFilingOutput` that passes citation-grounding validation.
- Prompt version constant lives in same file as prompt text; registered in Langfuse.
- Content-hash cache hit returns identical output on rerun without LLM call.
- A corrupted citation in test fixture routes to `data/quarantine/s_filings/`.
- `bump-prompt-version` skill checks pass.

**Labels.** `extract`, `m` **Milestone.** W1 **Blocked by.** #5, #9, #10

### Issue 12 — `feat(cli): CLI entry point + W1 acceptance smoke`

**Objective.** Click-based `auto-research` CLI: `ingest edgar`, `ingest fmp`,
`extract s-filings`, `feast apply`, `feast materialize`, `eval extract`,
`status`. End-to-end smoke: fetch one S-3, extract, populate Feast.

**Acceptance criteria.**
- `uv run auto-research status` prints Langfuse / MLflow / Feast registry health.
- End-to-end smoke (one ticker, one S-3) completes in `< 5 min` locally.
- `auto-research --help` documents every subcommand and required env var.

**Labels.** `infra`, `m` **Milestone.** W1 **Blocked by.** #5, #6, #7, #11

---

## Milestone 2 — W2: RAG layer + extraction quality

### Issue 13 — `feat(extract): unstructured.io parsing + section-aware chunking`

**Objective.** `extract/chunking.py` parses 10-K via unstructured.io
respecting Item 1A / 7 / 8 boundaries. Output is `list[Chunk]` with
`section_name`, `char_span`, `token_count`.

**Acceptance criteria.**
- A real 10-K fixture parses into sections; assertions on section count + names.
- Chunks under 4K tokens; boundary-respecting (no chunk spans a section break).
- Test verifies `source_text[chunk.char_span] == chunk.text` (citation-friendly).

**Labels.** `rag`, `m` **Milestone.** W2 **Blocked by.** #1

### Issue 14 — `feat(extract): contextual chunking (Anthropic pattern)`

**Objective.** For each chunk, generate a one-line context (e.g.,
*"This chunk is from NVDA Q3-2025 10-Q MD&A discussing China export
controls"*) via cached LLM call and prepend to chunk text before embedding.

**Acceptance criteria.**
- Context generation calls use prompt caching (verified via VCR).
- Generated context is ≤ 100 tokens.
- Stored alongside chunk for audit.
- `bump-prompt-version` skill applied if the context prompt changes.

**Labels.** `rag`, `extract`, `m` **Milestone.** W2 **Blocked by.** #10, #13

### Issue 15 — `feat(extract): LanceDB + Voyage embeddings adapter (BGE local fallback)`

**Objective.** Embedding adapter wraps Voyage `voyage-3` (primary) with
`bge-small-en-v1.5` local fallback when `VOYAGE_API_KEY` absent or quota
exceeded. Persist per-doc LanceDB store at `data/rag/{doc_id}.lance`.

**Acceptance criteria.**
- Adapter unit tests cover both backends (Voyage VCR + BGE in-process).
- Same query against the same store returns deterministic top-k order.
- Fallback decision logged with reason (no key / quota / explicit override).

**Labels.** `rag`, `m` **Milestone.** W2 **Blocked by.** #14

### Issue 16 — `feat(extract): hybrid retrieval (BM25 + dense + RRF)`

**Objective.** `extract/rag_retrieval.py` runs BM25 (`rank_bm25`) and dense
retrieval in parallel and merges via Reciprocal Rank Fusion.

**Acceptance criteria.**
- Hybrid retrieve returns ranked candidates with per-source scores.
- Property test: RRF score is monotonic in component ranks.
- Hand-built micro-corpus test verifies RRF beats either retriever alone on
  precision@5 for at least 2 of 3 example queries.

**Labels.** `rag`, `m` **Milestone.** W2 **Blocked by.** #15

### Issue 17 — `feat(extract): BGE reranker on top-20 → top-5`

**Objective.** Add `bge-reranker-base` (local) reranking pass on hybrid
output before extraction.

**Acceptance criteria.**
- Rerank reorders the top-20 deterministically on the same input.
- Hand-built test: rerank improves precision@5 over RRF alone on micro-corpus.
- CPU-only inference; no GPU assumption.

**Labels.** `rag`, `s` **Milestone.** W2 **Blocked by.** #16

### Issue 18 — `feat(extract): entity resolution (Flow 3 — supplier mention → ticker)`

**Objective.** `extract/entity_resolution.py` maps fuzzy supplier mentions
to tradeable tickers. Universe + aliases embedded once with Voyage; mention
text → top-3 candidates → LLM disambiguator picks or returns `unknown`.

**Acceptance criteria.**
- Hand-built gold set of ~20 mentions; F1 ≥ 0.85 after reranker.
- Disambiguator stores reasoning per resolution for audit (DeepEval reads).
- `unknown` is an allowed output (no false-confident matches).

**Labels.** `rag`, `m` **Milestone.** W2 **Blocked by.** #15

### Issue 19 — `feat(extract): 10-K, transcript, 8-K worker bodies + prompts`

**Objective.** Implement the remaining three extraction workers and their
prompts. 10-K uses contextual-RAG path for docs ≥ 100K tokens; others
single-shot with caching.

**Acceptance criteria.**
- Each worker produces a frozen output passing citation grounding on a
  real fixture.
- Each prompt has its own version constant; all registered in Langfuse.
- Hybrid extraction policy: single-shot for `< 100K` tokens (verified by
  branch coverage); RAG path for `≥ 100K`.
- `bump-prompt-version` skill applied.

**Labels.** `extract`, `l` **Milestone.** W2 **Blocked by.** #11, #14, #17, #18

### Issue 20 — `feat(eval): gold sets + DeepEval pytest harness`

**Objective.** Hand-label ~50-80 examples per worker at
`eval/gold_sets/{worker}.jsonl`. Build DeepEval pytest suite per
`docs/specs/2026-05-22-design.md` §14.1: F1 / exact-match / Spearman per
field, G-Eval for subjective fields, hallucination metric.

**Acceptance criteria.**
- Gold sets committed under `eval/gold_sets/` (size noted per worker).
- `tests/evals/test_{worker}_extraction.py` runs under DeepEval pytest.
- Baselines captured at `eval/baselines/{worker}__{prompt_version}__*.json`
  for the current prompt versions.
- Suite passes the published baseline thresholds (otherwise issue calls
  out the gap for fix in #22).

**Labels.** `eval`, `l` **Milestone.** W2 **Blocked by.** #19

### Issue 21 — `feat(eval): Ragas RAG eval (context_recall, faithfulness, answer_relevancy)`

**Objective.** Hand-build ~30 `(query, expected_chunk_ids)` pairs for Flow 1
and ~30 `(query, expected_memo_ids)` pairs for Flow 2. Build Ragas pytest
suite per spec §8.5.

**Acceptance criteria.**
- Eval sets committed; size + composition documented in test docstrings.
- `tests/evals/test_rag_*.py` runs under Ragas pytest.
- Flow 2 baseline meets `context_recall > 0.75` and `faithfulness > 0.85`
  (or the gap is documented + tracked).

**Labels.** `eval`, `rag`, `m` **Milestone.** W2 **Blocked by.** #16, #17, #18

### Issue 22 — `feat(extract): backfill orchestrator + Anthropic Batch API kickoff`

**Objective.** Build the 2-year backfill orchestrator and kick off the
Anthropic Batch API run for ~2,700 docs across 90 names. **Live spend
~$75-150 — requires explicit user confirmation before launching the batch**
(AGENTS.md §5 prohibition).

**Acceptance criteria.**
- `auto-research extract backfill --dry-run` prints estimated cost +
  doc count + per-tier model breakdown without API calls.
- `--confirm` flag required to submit the actual Batch API job.
- Batch job ID + estimated completion time logged to MLflow.
- Resumable: a partial-failure run continues from manifest state.
- Post-batch run populates Feast via `feast materialize-incremental`.

**Labels.** `extract`, `infra`, `l` **Milestone.** W2 **Blocked by.** #19, #20

---

## Milestone 3 — W3: Signals + backtest gauntlet

### Issue 23 — `feat(backtest): T1 info_tests primitives` [SENSITIVE]

**Objective.** Implement `backtest/info_tests.py` per `docs/BACKTEST.md` §3:
`event_study`, `ic_analysis`, `quantile_sort`, `conditional_distribution`,
`mutual_information`, `bootstrap_significance`, plus `InfoReport` dataclass.

**Acceptance criteria.**
- Each primitive has a unit test against a synthetic dataset with a known
  ground-truth effect.
- `InfoReport` is frozen.
- Property test: `bootstrap_significance` CI width shrinks as `n_boot` grows.
- All numeric stats include t-stat + n.

**Labels.** `backtest`, `m` **Milestone.** W3 **Blocked by.** #7

### Issue 24 — `feat(backtest): triple-barrier labels + CPCV + deflated Sharpe + cost model + reports` [SENSITIVE]

**Objective.** Implement the López de Prado primitives per
`docs/BACKTEST.md` §4: `labels.triple_barrier_label`,
`cpcv.cpcv_splits`, `deflated_sharpe.deflated_sharpe`,
`costs.realistic_costs`, plus `BacktestReport` dataclass.

**Acceptance criteria.**
- Triple-barrier: vol-adjusted bands; property test on barrier-hit ordering.
- CPCV property tests (per BACKTEST.md §4.2) verify: no train/test overlap,
  embargo respected, every sample appears in test the right number of times.
- Deflated Sharpe regression test against published example values.
- Cost model returns a typed `CostBreakdown`; plumbed via
  `vbt.Portfolio.from_signals` smoke test.
- `BacktestReport.sharpe_gross_diagnostic_only` exists but
  `tests/backtest/test_no_gross_sharpe_in_gates.py` asserts no gate code reads it.

**Labels.** `backtest`, `l` **Milestone.** W3 **Blocked by.** #23

### Issue 25 — `feat(backtest): vbt.pro engine wrapper + tier-aware gates` [SENSITIVE]

**Objective.** `backtest/engine.py` wraps vbt.pro for the T2 portfolio
backtest per `docs/BACKTEST.md` §4.5. `backtest/gates.py` defines
`T1_GATE` + `T2_GATE` constants (verbatim from BACKTEST.md §2) and the
`check_t1_gate` / `check_t2_gate` functions.

**Acceptance criteria.**
- `T1_GATE` / `T2_GATE` constants match `docs/BACKTEST.md` §2 verbatim.
- `check_t1_gate(report)` returns `"promote"` iff every threshold is
  met (unit-tested at boundaries).
- Engine smoke test on a synthetic signal completes end-to-end (CPCV →
  cost-adjusted Sharpe → deflated → gate decision).
- Static test (`tests/backtest/test_no_llm_in_gates.py`) verifies neither
  gate function imports any LLM client.

**Labels.** `backtest`, `m` **Milestone.** W3 **Blocked by.** #24

### Issue 26 — `feat(signals): A2 + A1 + B1 + IC-weighted combiner + AlphaLibrary`

**Objective.** Build signals A2 (PEAD language drift), A1 (supply-chain
forward-tone), B1 (frontier milestone/dilution) per spec §9. IC-weighted
combiner with Ledoit-Wolf shrinkage. `AlphaLibrary` backed by MLflow runs
per `docs/CONTRACTS.md`.

**Acceptance criteria.**
- Each signal runs end-to-end (Feast → score → T1 → T2 → BacktestReport).
- A2 promoted to alpha library iff T2_GATE passes (cite the report).
- A1 promoted iff T2_GATE passes (cite the report).
- B1 outcome (promote or kill) documented either way — kill memos count.
- Combiner returns per-name daily alpha; covariance regularization unit-tested.
- AlphaLibrary registry round-trips via MLflow API.

**Labels.** `signal`, `backtest`, `l` **Milestone.** W3 **Blocked by.** #25, #18, #19

---

## Milestone 4 — W4: Research agent + live critic + MCP + polish

### Issue 27 — `feat(mcp): FastMCP server exposing read-only data + research interface` [SENSITIVE]

**Objective.** Implement `src/auto_research/mcp_server.py` per
`docs/CONTRACTS.md` §2: `query_features`, `run_backtest`, `search_memos`,
`list_alpha_library`, `read_signal_performance`, `get_feature_definition`.
Read-only contract; no write tools.

**Acceptance criteria.**
- Every tool has Pydantic input + output models, no raw dicts.
- Unit test per tool exercises the end-to-end path with `TestClient`,
  no external service.
- Smoke test against the live in-process server runs all tools without error.
- Static test asserts no tool function mutates state outside MLflow / Feast read paths.
- README documents Claude Desktop wiring via `.cursor/mcp.json` /
  Claude config `mcp.json`.

**Labels.** `mcp`, `agent`, `l` **Milestone.** W4 **Blocked by.** #7, #25, #26

### Issue 28 — `feat(agents): LangGraph research agent — state + checkpointer + HITL interrupt` [SENSITIVE]

**Objective.** Build the research graph per `docs/CONTRACTS.md` §3 and spec
§11: `ResearchState`, `HypothesisType` enum, six nodes
(propose_hypothesis → materialize_signal → run_validation → decide ‖
critique → write_memo), `SqliteSaver` checkpointer, `interrupt()` at
`decide_promote`.

**Acceptance criteria.**
- Graph compiles and runs end-to-end with a `FakeListLLM` for the
  hypothesis-proposing node.
- `decide` reads only `state.validation_report`; `critique` writes only
  `state.critique`; static test verifies these.
- Checkpointer resumes a graph mid-execution (test on a deliberately
  killed session).
- HITL interrupt fires before any signal is added to the alpha library.
- Cost-cap + circuit-breaker + max-iterations decorators applied to every
  LLM-touching node.

**Labels.** `agent`, `l` **Milestone.** W4 **Blocked by.** #26, #27

### Issue 29 — `feat(agents): Pydantic AI live critic + two worked examples`

**Objective.** Implement the daily live critic per `docs/CONTRACTS.md` §4.
Worked examples: one signal that gets promoted, one that gets killed —
both produce attribution memos demonstrating the haircut output.

**Acceptance criteria.**
- `LiveCriticInput` / `LiveCriticOutput` Pydantic models with
  schema-enforced `haircut ∈ [0, 1]` (1.0 valid, > 1.0 invalid).
- Unit tests use Pydantic AI's `TestModel` (no API call).
- Two end-to-end demo scripts produce signed haircut memos under `data/memos/`.
- `scripts/cron_daily_critic.sh` documented in README.

**Labels.** `agent`, `m` **Milestone.** W4 **Blocked by.** #26

### Issue 30 — `docs: README rewrite + architecture diagram + signal cards + Langfuse trace links`

**Objective.** Rewrite README in the PDF-v0-critique framing per spec §1
and §20. Add signal cards (`docs/signal_cards/{A2,A1,B1}_*.md`) — logic,
why it works, validation results, cuts-list position. Add public Langfuse
trace links demonstrating the research-agent loop end-to-end.

**Acceptance criteria.**
- README under 200 lines; opens with the v0-critique thesis; covers
  architecture, two-plane diagram, MCP server wiring, quickstart, status.
- Three signal cards committed.
- At least one public Langfuse trace URL embedded in README (research-agent
  end-to-end successful session).
- Cuts list (spec §19) visible in README "Status" section.

**Labels.** `docs`, `m` **Milestone.** W4 **Blocked by.** #26, #28, #29

### Issue 31 — `chore(repo): scripts/setup_github.sh + scripts/create_all_issues.py`

**Objective.** **Run by hand first as the bootstrap** (creates repo,
milestones, labels, Issues 1-30 + 32 from this plan). The implementation
ships the two scripts as a committed PR closing Issue 31.

**Acceptance criteria.**
- `scripts/setup_github.sh` idempotent; creates 4 milestones, 11 labels (per
  conventions section above), 4 size labels.
- `scripts/create_all_issues.py` parses this plan file, extracts AC, infers
  label set from per-issue label hints, posts via `gh issue create`.
- Smoke test parses this plan and reports 32 issues found.
- Re-running both scripts is a no-op (title-based dedup).

**Labels.** `infra`, `polish`, `s` **Milestone.** W4 **Blocked by.** —

### Issue 32 — `feat(dashboard): optional Streamlit dashboard` (stretch; first cut candidate)

**Objective.** `dashboard.py` reads from MLflow + DuckDB + Feast and shows
live PnL chart, per-signal IC time-series, current critic haircuts, alpha
library status, recent attribution memos. Per spec §17, **first cut** if W4
buffer compresses.

**Acceptance criteria.**
- `uv run streamlit run dashboard.py` boots without error against an empty
  MLflow / Feast.
- All panels degrade gracefully when data is missing.
- Listed as `extension` in README, not a core component.

**Labels.** `polish`, `m` **Milestone.** W4 **Blocked by.** #26, #29

---

## Done definition for the whole plan

The v1 build is done when:

- All sensitive paths marked `[SENSITIVE]` above pass their named test/property test.
- At least one signal lives in the alpha library with a complete
  `BacktestReport` (deflated Sharpe, net-of-cost, CPCV indices logged).
- One worked research-agent session has produced a memo end-to-end.
- One worked live-critic run has produced a signed haircut.
- MCP server is wired into Claude Desktop / Cursor and queries return results.
- README opens with the PDF-v0-critique thesis and shows ≥ 1 public Langfuse trace.
- 32/32 issues closed; 32 PRs in the repo, all mapping to AC; cuts taken
  documented in README "Status."
