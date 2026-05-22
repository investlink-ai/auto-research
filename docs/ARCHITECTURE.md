# Architecture

Current architecture. Updated as code evolves.

For design rationale and the original v1 narrative, see
`docs/specs/2026-05-22-design.md`. The spec is a frozen design record; this
doc is the live map.

---

## 1. Two-plane overview

```
        ┌────────────────────────────────────────────┐
        │             DATA SOURCES                     │
        │   EDGAR  •  FMP transcripts  •  prices       │
        └─────────────────┬──────────────────────────┘
                          ▼
        ┌────────────────────────────────────────────┐
        │             RAW DOC STORE                    │
        │   data/raw/  (content-hash idempotent)       │
        └──────┬─────────────────────────┬───────────┘
               │                         │
       LLM PLANE                  DETERMINISTIC PLANE
   ┌─────────────────┐         ┌──────────────────────┐
   │ extract/        │         │ ingest/ → prices     │
   │ (workers, RAG,  │         │ feast/  (PIT FS)     │
   │  guardrails)    │ ──────▶ │                      │
   └─────────────────┘ features└──────────┬───────────┘
                                          ▼
                                ┌──────────────────────┐
                                │ signals/  + combiner │
                                └──────────┬───────────┘
                                           ▼
                                ┌──────────────────────┐
                                │ backtest/   tiered   │
                                │  T1 → T2 → T3 gates  │
                                └──────────┬───────────┘
                                           ▼
                                ┌──────────────────────┐
                                │ paper portfolio +    │
                                │ MLflow alpha library │
                                └──────────┬───────────┘
                                           │
   ┌───────────────────────────────────────┘
   ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│ agents/research_graph    │ ──▶ │ mcp_server (FastMCP)     │ ◀── Cursor / Claude Desktop
│  (LangGraph: hypothesis  │ ◀── │  read-only tool surface  │
│   → validate → memo)     │     └──────────────────────────┘
└──────────────────────────┘
┌──────────────────────────┐
│ agents/live_critic       │ ─── daily haircut ∈ [0, 1]
│  (Pydantic AI)           │
└──────────────────────────┘
```

The **Feast feature store** is the only contract between the two planes.

---

## 2. Plane boundaries

### Deterministic plane

Owns prices, features, signals, backtests, portfolio state. No LLM calls in
the trading-decision path. Code-checked gates (`T1_GATE`, `T2_GATE`) decide
promote/iterate/kill. See `docs/BACKTEST.md`.

### LLM plane

Owns unstructured-to-structured extraction (nightly batch), research
hypotheses (async), live haircut critic (multiplicative only). All LLM calls
are version-pinned, content-hash cached, and produce typed Pydantic outputs
with citation grounding. See `docs/CONTRACTS.md`.

### Boundary contract

Features written to Feast carry `event_datetime` + `as_of_ts = event_datetime + 1
trading day cutoff`, baked in at write-time (INV-1). The deterministic plane
reads Feast only; the LLM plane writes Feast only. No raw extraction JSON
crosses the boundary into signals — that would bypass PIT discipline.

---

## 3. Module map

```
src/auto_research/
├── ingest/                   # data sources
│   ├── edgar.py              # SEC EDGAR client (10-K, 10-Q, 8-K, S-1, S-3)
│   ├── fmp.py                # FMP transcripts + bid-ask half-spread
│   └── manifest.py           # append-only fetch ledger
│
├── extract/                  # LLM plane
│   ├── schemas.py            # Pydantic outputs with source_span/source_quote      [SENSITIVE]
│   ├── guardrails.py         # citation grounding + Guardrails AI validators       [SENSITIVE]
│   ├── chunking.py           # unstructured.io + contextual chunking
│   ├── rag_retrieval.py      # LanceDB + hybrid (BM25+dense+RRF) + BGE reranker
│   ├── entity_resolution.py  # mention → ticker disambiguation
│   ├── ten_k.py              # 10-K worker
│   ├── transcript.py         # earnings transcript worker
│   ├── eight_k.py            # 8-K worker
│   └── s_filings.py          # S-1, S-3 worker
│
├── feast_repo/               # PIT feature store                                    [SENSITIVE]
│   ├── feature_store.yaml    # registry + offline store config
│   ├── entities.py           # entity_id = ticker
│   ├── feature_views.py      # ten_k, transcript, eight_k, s_filing, price, signal
│   └── feature_services.py   # signal_a1, signal_a2, signal_b1
│
├── signals/                  # alpha primitives
│   ├── a1_supply_chain.py    # hyperscaler forward-tone propagation
│   ├── a2_pead.py            # post-earnings language drift
│   ├── b1_frontier.py        # milestone/dilution
│   └── combiner.py           # IC-weighted + Ledoit-Wolf shrinkage
│
├── backtest/                 # validation gauntlet                                  [SENSITIVE]
│   ├── info_tests.py         # T1: event_study, ic_analysis, quantile_sort, MI
│   ├── labels.py             # triple-barrier with vol-adjusted bands              [SENSITIVE]
│   ├── cpcv.py               # combinatorial purged CV with embargo                [SENSITIVE]
│   ├── deflated_sharpe.py    # multiple-testing-adjusted Sharpe                    [SENSITIVE]
│   ├── costs.py              # half-spread + sqrt impact + borrow + commissions    [SENSITIVE]
│   ├── engine.py             # vbt.pro wrapper, T1/T2 tier dispatcher
│   ├── report.py             # InfoReport, BacktestReport dataclasses
│   └── gates.py              # T1_GATE, T2_GATE constants (code-checked)           [SENSITIVE]
│
├── agents/                   # LLM-driven workflows
│   ├── research_graph.py     # LangGraph state machine + checkpointer + HITL       [SENSITIVE]
│   ├── live_critic.py        # Pydantic AI daily haircut
│   ├── memo_retrieval.py     # Flow 2 RAG over past memos
│   ├── reliability.py        # circuit breaker, cost cap, fallback model           [SENSITIVE]
│   └── alpha_library.py      # MLflow-backed promoted-signal registry
│
├── mcp_server.py             # FastMCP read-only tool surface                       [SENSITIVE]
│
└── eval/                     # quality gates
    ├── deepeval_suite.py     # extraction F1, hallucination, G-Eval
    └── ragas_suite.py        # RAG context_recall, faithfulness
```

`[SENSITIVE]` marks Tier 2 paths per `docs/AI_WORKFLOW.md` §2. Edits require
failing test or eval delta first.

---

## 4. Data flow per layer

| Layer | Reads from | Writes to | Cadence |
|---|---|---|---|
| **Ingest** | EDGAR, FMP, price API | `data/raw/`, `data/manifest.parquet` | Nightly cron |
| **Extract** | `data/raw/` | `data/extracted/*.jsonl`, Feast FeatureViews | Nightly batch (Anthropic Batch API) |
| **Feast materialize** | `data/extracted/`, prices | Feast offline store (Parquet) | After each extract run |
| **Signals** | Feast (read-only) | `signal_features` FeatureView | Daily |
| **Backtest** | Feast (`signal_features` + prices) | `MLflow runs`, `BacktestReport` artifacts | On-demand (research) + weekly batch |
| **Research agent** | MCP tools → Feast/MLflow/memos | `data/memos/*.md`, `alpha_library` (MLflow) | Async, hours-to-days |
| **Live critic** | News API, current positions | Daily `haircut.json` consumed by paper portfolio | Daily cron |

---

## 5. Ownership boundaries

| Concern | Owner | Not owned by |
|---|---|---|
| Point-in-time correctness | `feast_repo/feature_views.py` (write-time baking) | Query callers |
| Citation grounding | `extract/guardrails.py` post-validation | Worker bodies |
| Promote/iterate/kill decision | `backtest/gates.py` (code constants) | Research agent LLM |
| Position state | Paper-portfolio engine (vbt.Portfolio) | Live critic, research agent |
| Prompt versioning | Langfuse prompt registry | Worker code (workers read registry) |
| Cost model | `backtest/costs.py` | Signal code, paper portfolio |
| MCP tool registration | `mcp_server.py` (read-only only) | Research agent (consumes, doesn't add) |

---

## 6. External services

| Service | Used by | Purpose |
|---|---|---|
| **Anthropic API** (Batch + caching) | `extract/`, `agents/` | Workers + LangGraph nodes + Pydantic AI critic |
| **FMP API** | `ingest/fmp.py`, `backtest/costs.py` | Transcripts + bid-ask half-spread |
| **Voyage AI** | `extract/rag_retrieval.py`, `extract/entity_resolution.py` | `voyage-3` embeddings (BGE fallback) |
| **EDGAR** | `ingest/edgar.py` | Free SEC filings |
| **Langfuse self-hosted** (Docker) | All LLM-touching code via OpenLLMetry | Traces, prompt registry, cost tracking |
| **MLflow local** | `backtest/`, `agents/alpha_library.py` | Experiment tracking + signal registry |
| **LanceDB local** | `extract/rag_retrieval.py`, `agents/memo_retrieval.py` | Vector store (per-doc + memos) |

---

## 7. Where to look when…

| Task | Start here |
|---|---|
| Add a new extraction field | `docs/CONTRACTS.md` §1 (schema), then `extract/{worker}.py` |
| Change a feature definition | `docs/DATA_MODEL.md`, then `feast_repo/feature_views.py` |
| Add or tune a signal | `docs/BACKTEST.md` §2 (tier gates), then `signals/` |
| Modify backtest math | `docs/BACKTEST.md` §3-5, then `backtest/{module}.py` |
| Add a research-agent tool | `docs/CONTRACTS.md` §2 (MCP surface), then `mcp_server.py` |
| Change LLM model routing | `docs/specs/2026-05-22-design.md` §7.3, then `extract/{worker}.py` |
| Debug a failing eval | `docs/AI_WORKFLOW.md` §5 (PR evidence template), then `eval/` |
