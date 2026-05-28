# XBRL SQL Layer — Queryable Structured Financial Data for the Research Agent

- Date: 2026-05-28
- Status: **Draft (brainstorming)**
- Owners: Sam
- Related: `docs/specs/2026-05-22-design.md` §6 (storage), `AGENTS.md` §2 (INV-1, INV-2), `docs/decisions/2026-05-25-foreign-filers-deferred.md`

## 1. Context

The current architecture (`docs/specs/2026-05-22-design.md` §6) stores
narrative content (10-K MD&A, transcripts, 8-K text) as embeddings in
LanceDB for hybrid retrieve+rerank. There is **no queryable store for
structured numerics** (revenue, segment breakdowns, capex, FCF, balance-
sheet items). The research agent today can ask "what did mgmt say about
data-center demand?" via RAG, but cannot ask "what was NVDA's data-
center segment revenue in Q2'25 vs Q2'24?" without re-extracting numbers
from narrative each time — a hallucination risk and a token waste.

Target state is **RAG + SQL** — two retrieval modalities exposed to the
research agent, each optimal for its query type. The narrative layer
already exists; this spec designs the SQL layer.

## 2. Goals

- **G1.** Provide a queryable structured-financials store sourced from
  SEC XBRL CompanyFacts JSON (free, regulator-published, citation-strong).
- **G2.** Agent-facing query surface is a small set of ergonomically-named
  wide views (`income_statement_arq`, `balance_sheet_mrq`, ...) the LLM
  can write SQL against with low error rate.
- **G3.** Source-faithful raw layer preserved underneath for restatement
  handling, PIT replay, and INV-2 citation.
- **G4.** PIT discipline (INV-1) baked at write-time: as-reported views
  return only facts knowable on a given as-of date.
- **G5.** Every fact carries an EDGAR accession number (`accn`) — the
  INV-2 citation bridge for any agent answer that quotes a number.
- **G6.** Schema and ingest interface designed for all-US-XBRL-era scope
  (~2009+, all `us-gaap` filers). Backfill is config-driven.

## 3. Non-goals

- **NG1.** No pre-XBRL (pre-2009) text-financials parsing. Different
  ingestion paradigm; deferred via separate ADR.
- **NG2.** No IFRS / foreign-filer support (20-F, 40-F) in v1. Extends
  the existing foreign-filers ADR; their fundamentals stay outside the
  XBRL store until a separate IFRS concept map ships.
- **NG3.** Not a replacement for Feast. The XBRL SQL store is an
  *upstream* of feature engineering and a *parallel* surface for the
  agent; Feast remains the only contract between the deterministic and
  LLM planes (per `AGENTS.md` §1).
- **NG4.** No real-time / intraday updates. Daily incremental backfill
  is sufficient — fundamentals change at filing cadence.

## 4. Industry references studied

Three schemas were studied before designing ours, to avoid re-deriving
solved problems:

### 4.1 SEC Financial Statement Data Sets (`sec.gov/files/aqfs.pdf`)

The regulator's own quarterly bulk dump. Four files, foreign-key joined:
`sub.tsv` (submissions, one per filing), `num.tsv` (numeric facts),
`tag.tsv` (XBRL taxonomy dictionary), `pre.tsv` (presentation order linking
tags to IS/BS/CF/EQ). The `num.tsv` model is:

```
(adsh, tag, version, coreg, ddate, qtrs, uom) → value
```

`qtrs` elegantly encodes period type in one integer: `0`=instant
(balance sheet), `1`=quarterly flow, `4`=annual flow. Restatements are
modeled by a new `adsh` (the 10-K/A or 10-Q/A) with the superseded
filing flagged `prevrpt=true` in `sub.tsv`. Taxonomy versioning lives in
`version` (e.g. `us-gaap/2024`).

This is the source-of-truth model for raw XBRL data, copied nearly
verbatim into our Layer 2.

### 4.2 Sharadar SF1 (Nasdaq Data Link)

One wide table keyed `(ticker, dimension, datekey)` with ~150 indicator
columns spanning IS/BS/CF/derived/market. The `dimension` column encodes
six PIT/restatement permutations in one categorical:

| dimension | meaning |
|---|---|
| `ARQ` / `ARY` / `ART` | As-Reported Quarterly / Annual / TTM |
| `MRQ` / `MRY` / `MRT` | Most-Recent Reported Quarterly / Annual / TTM |

AR-* rows are immutable point-in-time. MR-* rows include restatements.
Auxiliary fields: `calendardate` (calendar-quarter-normalized
for cross-sectional alignment), `datekey` (filing date — PIT cutoff),
`reportperiod` (actual fiscal period end), `lastupdated` (vendor
revision marker, distinct from filer restatement), `permaticker`
(stable issuer ID).

This is the elegant query surface model. We adopt the AR/MR split as
**separate views** (not a categorical column) for clarity and indexability.

### 4.3 Financial Modeling Prep (FMP) fundamentals API

Per-statement endpoints (`/income-statement/{ticker}`, ...) with a fixed
~30-50 column standardized schema, plus parallel `-as-reported` endpoints
returning raw XBRL tags. **No restatement model** — the API always
returns FMP's current canonical value, silently overwriting prior
snapshots. Field typos (`fillingDate`, `otherInvestingActivites`) are
stable in the schema — a cautionary tale about vendor lock-in on critical
interpretive layers.

Useful as a parallel ground-truth check during eval, not as a primary
source.

### 4.4 What we copy from each

| From | What we copy |
|---|---|
| SEC FSDS | Long fact-table shape, `(accn, tag, taxonomy, version, coreg, period_end, qtrs, unit, value)`; the `qtrs` period encoding; separate `submissions` and `tags` tables; `prevrpt` restatement flag |
| Sharadar | AR-vs-MR split (as separate views); `lastupdated` ingest-time marker; permanent issuer ID concept (but using SEC's CIK rather than inventing our own) |
| FMP | Negative lesson: snapshot every fetch ourselves; never trust a vendor's "current value" call as truth |

## 5. Architecture

Three layers, mirroring the dbt raw → staging → mart pattern:

```
Layer 3 (Mart, ergonomic):   income_statement_arq, income_statement_mrq,
                             balance_sheet_arq, balance_sheet_mrq,
                             cash_flow_arq, cash_flow_mrq
                             [+ _ary, _mry, _art, _mrt variants]
                                       ▲
                                       │  CREATE VIEW (concept-map-driven pivot)
Layer 2 (Staging, normalized):
                             xbrl_facts (long)
                             submissions (one row / filing)
                             tags        (XBRL taxonomy dictionary)
                             presentation (tag → IS/BS/CF)
                             tickers     (CIK ↔ ticker history)
                                       ▲
                                       │  per-CIK ingest worker
Layer 1 (Raw):               data/raw/xbrl/companyfacts/CIK{cik}.json
```

## 6. Layer 2 schema (source-faithful, normalized)

### 6.1 `xbrl_facts` — the long fact table

```
ticker         TEXT          -- denormalized for query speed
cik            BIGINT        -- permanent issuer id (SEC-assigned)
accn           TEXT          -- 20-char EDGAR accession; FK → submissions
tag            TEXT          -- e.g. 'Revenues', 'DataCenterRevenue'
taxonomy       TEXT          -- 'us-gaap', 'ifrs-full', 'nvda' (custom)
version        TEXT          -- '2024', or accn for custom extension
period_end     DATE          -- fiscal period end (ddate equivalent)
qtrs           SMALLINT      -- 0=instant, 1=Q-flow, 4=annual
unit           TEXT          -- 'USD', 'shares', 'pure'
value          DECIMAL(28,4)
coreg          TEXT          -- subsidiary/parent segment; NULL=consolidated
lastupdated    TIMESTAMP     -- our ingest timestamp
```

Primary key: `(accn, tag, taxonomy, version, coreg, period_end, qtrs, unit)`.

Partitioned by `EXTRACT(YEAR FROM submissions.filed)` for PIT-historical
query locality.

### 6.2 `submissions` — one row per XBRL filing

```
accn           TEXT PRIMARY KEY
cik            BIGINT
form           TEXT     -- '10-K', '10-Q', '10-K/A', '10-Q/A', '8-K'
period         DATE     -- balance-sheet date this submission reports
fy             SMALLINT -- fiscal year focus
fp             TEXT     -- 'FY', 'Q1'..'Q4', 'H1', 'H2', 'CY'
filed          DATE     -- SEC filing date
accepted       TIMESTAMP -- SEC acceptance datetime (the PIT stamp)
prevrpt        BOOLEAN  -- TRUE if a later /A amendment superseded this
instance       TEXT     -- XBRL instance document filename
```

`accepted` is the canonical event time. `as_of_ts = accepted + lag-1
trading day cutoff` per INV-1.

### 6.3 `tags` — XBRL taxonomy dictionary

```
tag            TEXT
taxonomy       TEXT
version        TEXT
datatype       TEXT     -- 'monetary', 'shares', 'pure'
iord           CHAR(1)  -- 'I' (instant) | 'D' (duration)
crdr           CHAR(1)  -- 'C' (credit) | 'D' (debit) — natural balance
abstract       BOOLEAN  -- non-numeric grouping tag
custom         BOOLEAN  -- TRUE = filer-specific extension
label          TEXT     -- preferred human-readable label
doc            TEXT     -- tag definition
```

Primary key: `(tag, taxonomy, version)`.

### 6.4 `presentation` — tag → statement type mapping

```
accn           TEXT     -- the filing this presentation applies to
report         INT
line           INT
stmt           CHAR(2)  -- 'BS', 'IS', 'CF', 'EQ', 'CI', 'UN', 'CP'
tag            TEXT
version        TEXT
plabel         TEXT
```

Sourced from `pre.tsv` in SEC FSDS bulk dumps for filings older than the
per-ticker CompanyFacts coverage; for current filings, derived from the
XBRL instance document.

### 6.5 `tickers` — CIK ↔ ticker history

```
cik            BIGINT
ticker         TEXT
valid_from     DATE
valid_to       DATE      -- NULL if currently active
```

Sourced from SEC `company_tickers.json` (current) + `submissions.cik`
historicals + `sub.former`/`sub.changed` for name-change events.
**CIK is the canonical entity key**; ticker joins are scoped by date.

## 7. Layer 3 — mart views (the agent's primary surface)

### 7.1 Concept map (`config/xbrl/concepts.yaml`)

The hand-maintained piece. For v1, ~30-50 concepts covering revenue,
COGS, gross profit, R&D, SG&A, opex, operating income, net income, EPS,
shares, cash, receivables, inventory, total current assets, PP&E,
goodwill, intangibles, total assets, accounts payable, debt (short /
long), total current liabilities, total liabilities, equity, operating
cash flow, capex, free cash flow, dividends paid.

```yaml
revenue:
  iord: D
  unit: USD
  tags:
    - { taxonomy: us-gaap, tag: Revenues }
    - { taxonomy: us-gaap, tag: SalesRevenueNet }
    - { taxonomy: us-gaap, tag: RevenueFromContractWithCustomerExcludingAssessedTax }
    - { taxonomy: us-gaap, tag: RevenueFromContractWithCustomerIncludingAssessedTax }
  overrides:
    NVDA: { taxonomy: us-gaap, tag: Revenues }

operating_income:
  iord: D
  unit: USD
  tags:
    - { taxonomy: us-gaap, tag: OperatingIncomeLoss }

# ...
```

Tag priority is ordered: first tag that has a value for the
`(ticker, period)` wins. Per-ticker `overrides` short-circuit the
priority list when a company uses an idiosyncratic standard or custom
extension tag.

### 7.2 View pattern

For each statement type × {AR, MR} × {Q, Y} = 12 views in v1. TTM
(`_art`, `_mrt`) variants deferred to v1.1 per §13. Pattern (simplified):

```sql
CREATE VIEW income_statement_arq AS
WITH ranked AS (
    SELECT
        ticker, cik, fy AS fiscal_year, fp AS fiscal_period, period_end,
        accepted, accn, concept, value, unit,
        ROW_NUMBER() OVER (
            PARTITION BY cik, period_end, concept
            ORDER BY accepted ASC   -- as-reported = FIRST known value
        ) AS rn
    FROM xbrl_facts_with_concept
    WHERE qtrs = 1
)
SELECT
    ticker, cik, fiscal_year, fiscal_period, period_end, accepted, accn,
    MAX(value) FILTER (WHERE concept = 'revenue')          AS revenue,
    MAX(value) FILTER (WHERE concept = 'cost_of_revenue')  AS cost_of_revenue,
    MAX(value) FILTER (WHERE concept = 'operating_income') AS operating_income,
    MAX(value) FILTER (WHERE concept = 'net_income')       AS net_income,
    ...
FROM ranked
WHERE rn = 1
GROUP BY ticker, cik, fiscal_year, fiscal_period, period_end, accepted, accn;
```

MR variant: `ORDER BY accepted DESC` (latest restated value wins).

Annual variants: `WHERE qtrs = 4`. TTM variants: derived from quarterly
rollup, separate materialization.

### 7.3 Why views, not materialized tables, for v1

DuckDB views are cheap; the long table is small enough (~3M rows for v1
universe; ~30M rows at full US-XBRL-era scope) that view execution stays
sub-second. We can promote any view to materialized table later if
profiling shows a hot spot.

## 8. PIT discipline (INV-1)

`as_of_ts = submissions.accepted + lag-1 trading day cutoff` baked into
the AR views at definition time. The AR view's `WHERE rn = 1 ORDER BY
accepted ASC` semantics guarantee no fact appears before its filing was
accepted by SEC.

**Only AR views are eligible inputs to Feast FeatureViews** — preserves
"Feast is the only contract between planes" (`AGENTS.md` §1) and INV-1's
"PIT lag-1 is structural, baked at write-time". MR views are agent-only.

Property test (`tests/xbrl/test_pit_properties.py`): for any
`(as_of, ticker, concept, period)`, the value returned by the AR view
filtered to `accepted <= as_of` must equal the value derivable from
re-reading the CompanyFacts JSON snapshot frozen at `as_of`.

## 9. Citation grounding (INV-2)

Every row in every view carries `accn`. New tool:

```
link_fact_to_filing(ticker, concept, period_end) →
    { accn, form, filed, accepted, edgar_url, document_url }
```

`edgar_url` and `document_url` are resolved via the existing ingest
manifest (`src/auto_research/ingest/manifest.py`) — the XBRL store reuses
the same source-of-truth registry as narrative ingestion. When the agent
says "revenue grew 26% YoY", it can cite the actual 10-Q, same INV-2
contract as narrative claims.

## 10. Storage engine

**DuckDB-over-Parquet.**

| Criterion | DuckDB | Postgres | SQLite |
|---|---|---|---|
| Infra cost | Zero (file-backed) | Docker service + migrations | Zero |
| Columnar / analytics | Native | Heavy plugins | No |
| LLM tool-use ergonomics | `duckdb.connect(read_only=True)` | Connection pool, auth | Driver overhead |
| Parquet integration | First-class (`read_parquet`) | Foreign data wrapper | Limited |
| Concurrent readers | Read-only mode OK | Native | OK |
| Scale headroom (30M rows) | Sub-second | Sub-second | Slower analytics |

Parquet partitioning by `submissions.filed` year survives PIT-historical
queries. DuckDB's read-only mode lets the agent's SQL tool and analyst
notebooks query concurrently without locking.

## 11. Ingest worker

`src/auto_research/ingest/xbrl/companyfacts.py` — new module, follows
existing `src/auto_research/ingest/_http.py` patterns (UA, fair-access
backoff, manifest append).

Per-CIK:

1. Resolve ticker → CIK via `tickers` table (refresh daily from
   `company_tickers.json`).
2. `GET data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json`.
3. Persist raw JSON to `data/raw/xbrl/companyfacts/CIK{cik:010d}.json`
   (immutable, content-hashed for cache).
4. Parse to long-form rows; populate `xbrl_facts`, `submissions`, and
   (incrementally) `tags`.
5. Append to ingest manifest with `event_datetime = submissions.accepted`
   for each new filing, `lastupdated = now()` on each fact row.
6. Idempotent: re-running for an unchanged JSON is a no-op via
   content-hash check against manifest.

Bulk historical bootstrap (one-time): consume SEC FSDS quarterly ZIPs
(2009-Q1 → present) to populate `presentation` and pre-fill the long
table for filings older than CompanyFacts coverage. After bootstrap,
incremental updates come from per-CIK CompanyFacts pulls.

CLI surface (extending existing `ingest` subcommands):

```
auto-research ingest xbrl backfill --universe v1
auto-research ingest xbrl refresh --since 2026-05-27
```

## 12. RAG + SQL integration

Three MCP / tool-use surfaces exposed to the LangGraph research agent:

| Tool | Purpose |
|---|---|
| `search_narrative(query, top_k)` | Existing LanceDB hybrid retrieve+rerank over 10-K / transcript / 8-K narrative chunks |
| `query_financials(sql)` | **New.** Read-only DuckDB query over the mart views; tool description ships wide-view schema + 5 example queries |
| `link_fact_to_filing(ticker, concept, period)` | **New.** INV-2 bridge — returns the source filing for any numeric SQL answer |

The agent routes naturally from the tool descriptions:

- "What was NVDA's data-center segment revenue in Q2'25?" → `query_financials`
- "What did Jensen Huang say about Blackwell production?" → `search_narrative`
- "Why did revenue beat consensus?" → both, then synthesize with citations

The `query_financials` tool description includes the wide-view DDL,
~30 concept-column names, and example queries. The agent does **not**
see the long fact table directly — keeps the SQL it generates tractable
and reduces LLM error rate.

## 13. Scope

| | v1 |
|---|---|
| Schema | All-US-XBRL-era (`us-gaap`, 2009+) — forward-compatible |
| Ingest interface | Universe-agnostic; backfill is config-driven |
| Backfill | 80-ticker `universe_v1` (matches existing extraction pipeline) |
| Concept map | ~30-50 concepts covering AI/semis/cloud reporting norms |
| Mart views | IS/BS/CF × AR/MR × Q/Y = 12 views; TTM (`+_art`, `+_mrt`) deferred to v1.1 |

### Deferred via separate ADRs

- **Pre-XBRL (pre-2009) text/PDF financial-statement parsing.** Separate
  ingestion paradigm, separate eval suite. Likely never needed for our
  use case but flagged for completeness.
- **IFRS / foreign-filer (20-F, 40-F) fundamentals.** Extends the
  existing `2026-05-25-foreign-filers-deferred.md` ADR. Their
  fundamentals require an `ifrs-full` concept map, not us-gaap.
- **XBRL dimension axes beyond `coreg`.** Segment disclosures (e.g. NVDA
  Data Center vs Gaming vs Auto) live in XBRL dimension axes
  (`us-gaap:StatementBusinessSegmentsAxis`). v1 uses CompanyFacts which
  exposes only the consolidated `coreg=NULL` slice; segment data
  requires either the richer FSNDS dataset or per-filing instance
  parsing. Deferred — flagged because segment data is high-value for
  AI/semis research and is the most likely v1.1 expansion.

## 14. Testing

| Suite | Purpose |
|---|---|
| `tests/xbrl/test_companyfacts_parser.py` | Pure-function tests: CompanyFacts JSON → long rows. Fixtures: NVDA, AAPL, banks (different statement shapes), an off-fiscal-calendar filer |
| `tests/xbrl/test_pit_properties.py` | Property test: AR view value at any as-of date equals value knowable from CompanyFacts JSON frozen at that date |
| `tests/xbrl/test_restatements.py` | Verifies MR view returns the latest restated value; AR view returns the originally-filed value; both reference correct `accn` |
| `tests/xbrl/test_concept_map_eval.py` | Eval suite: for each `(ticker, concept, period)` in a hand-curated golden table (sourced from the actual 10-K), the wide-view value matches. Runs on every concept-map config change. |
| `tests/xbrl/test_citation_resolution.py` | `link_fact_to_filing` returns a valid EDGAR URL for every `accn` |
| `tests/live/test_xbrl_ingest.py` | Live SEC API call, gated behind `RUN_LIVE_TESTS=1` per repo test taxonomy |

## 15. Open questions

- **Q1.** Does the agent need a separate `as_of` parameter in
  `query_financials` to back-date queries (e.g. "what would I have known
  about NVDA on 2024-03-15"), or is "current MR view" sufficient for v1?
  Recommend deferring `as_of` parameter to v1.1.
- **Q2.** Should we expose the long `xbrl_facts` table to the agent as
  a fallback when a concept isn't in the map? Recommend **no** for v1
  — every concept the agent needs should go through the eval-tested
  map; un-mapped queries should fail loudly so we add the concept
  rather than silently get the wrong tag.
- **Q3.** TTM materialization — derived view over quarterly rollup, or
  separate ingest-time materialization? Recommend derived view for v1
  (consistency with `_arq`/`_mrq` pattern); revisit if query latency
  matters.

## 16. Rollback

Trivial: drop the `xbrl_facts` schema and views from DuckDB, remove
`config/xbrl/concepts.yaml`, remove the `ingest xbrl` CLI subcommand,
remove the three new tools from the agent's MCP surface. The existing
Feast + LanceDB stack is untouched throughout.
