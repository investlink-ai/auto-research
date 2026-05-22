# Contracts

Stable interfaces. Re-read this before changing any Pydantic schema, MCP tool,
or domain dataclass — these are the boundaries other modules depend on.

For runtime architecture see `docs/ARCHITECTURE.md`. For data semantics see
`docs/DATA_MODEL.md`.

---

## 1. Extraction schemas

All extraction workers return `BaseModel` subclasses from
`src/auto_research/extract/schemas.py`. Every claim carries
`source_span: tuple[int, int]` and `source_quote: str`. The
`citation_grounding_validator` in `extract/guardrails.py` asserts
`source_text[span[0]:span[1]] == source_quote` for every claim before
persistence. Failures route to `data/quarantine/{worker}/{doc_id}.json`.

### 1.1 Base types

```python
class Citation(BaseModel):
    source_span: tuple[int, int]
    source_quote: str  # verbatim slice of source_text

class Claim(BaseModel):
    citation: Citation
    confidence: float = Field(ge=0.0, le=1.0)
```

Every domain claim composes a `Claim`. Don't add a claim type that bypasses
this composition; the post-validator depends on it.

### 1.2 Per-worker outputs

```python
# extract/schemas.py

class TenKOutput(BaseModel):
    cik: str
    accession_number: str
    fiscal_period_end: date
    guidance_tone: Claim                        # subjective; G-Eval scored
    accrual_flags: list[Claim]
    supplier_mentions: list[SupplierMention]    # entity-resolved later
    customer_mentions: list[CustomerMention]
    language_novelty_score: float               # vs prior 10-K, computed
    risk_factor_deltas: list[RiskFactorDelta]

class TranscriptOutput(BaseModel):
    ticker: str
    event_datetime: datetime
    prepared_remarks_tone: Claim
    q_and_a_evasiveness: Claim                  # subjective; G-Eval scored
    forward_statements: list[ForwardStatement]  # entity links + horizons

class EightKOutput(BaseModel):
    cik: str
    accession_number: str
    event_classification: Literal[
        "milestone", "partnership", "contract", "guidance_change",
        "leadership_change", "dilution", "other"
    ]
    milestone_mentions: list[Claim]
    dilution_language_flags: list[Claim]

class SFilingOutput(BaseModel):
    cik: str
    accession_number: str
    form_type: Literal["S-1", "S-3"]
    dilution_event: Claim
    capital_raise_language: list[Claim]
    use_of_proceeds: list[Claim]

class SupplierMention(BaseModel):
    mention_text: str
    citation: Citation
    resolved_ticker: str | None                 # None until entity resolution
    resolver_confidence: float | None
    resolver_reasoning: str | None
```

### 1.3 Contract rules

- All output models are **frozen** (`model_config = ConfigDict(frozen=True)`).
- All fields are explicit; no `Optional` defaults except where genuinely
  optional (e.g., `resolved_ticker` before entity resolution runs).
- Adding a field is non-breaking; removing or renaming requires a
  `prompt_version` bump in Langfuse (see INV-6) and a Feast schema migration.
- Worker functions are `(raw_doc: RawDoc, prompt_version: str) → Output` —
  pure functions. Content-hash cached on `sha256(raw_doc.bytes + prompt_version)`.

---

## 2. MCP tool surface

`src/auto_research/mcp_server.py` exposes the read-only research interface.
Consumed by the in-process LangGraph research agent and external clients
(Claude Desktop, Cursor MCP client). **Read-only.** No tool may mutate state.

```python
@mcp.tool()
def query_features(
    entity_df: pd.DataFrame,           # cols: entity_id, event_timestamp
    feature_refs: list[str],           # e.g. ["transcript_features:q_and_a_evasiveness"]
) -> pd.DataFrame: ...
# PIT-correct join via Feast.get_historical_features

@mcp.tool()
def run_backtest(
    signal_def: SignalDefinition,      # pickled signals.SignalDefinition
    params: BacktestParams,
    tier: Literal["T1", "T2", "T3"],
) -> InfoReport | BacktestReport: ...
# Returns the appropriate report type; tier-routed

@mcp.tool()
def search_memos(
    query: str,
    k: int = 5,
    filter_status: Literal["promoted", "killed", "iterated", None] = None,
) -> list[MemoHit]: ...
# Flow 2 RAG: BM25 + dense + RRF + BGE reranker

@mcp.tool()
def list_alpha_library() -> list[AlphaLibraryEntry]: ...
# Reads MLflow registry

@mcp.tool()
def read_signal_performance(
    signal_id: str,
    window: tuple[date, date],
) -> SignalPerformance: ...
# IC time-series + PnL attribution from MLflow

@mcp.tool()
def get_feature_definition(
    feature_view: str,
    feature_name: str,
) -> FeatureDefinition: ...
# Reads from Feast registry
```

### 2.1 Tool surface rules

- **Read-only.** No `promote_signal`, no `update_universe`, no `write_memo`.
  Mutations belong to the LangGraph agent's internal nodes, not to tools.
- **Idempotent.** Same input → same output (modulo timestamp on
  `read_signal_performance` end-window).
- **Typed.** Every tool's inputs and outputs are Pydantic models. No raw dicts.
- **Cost-bounded.** Each tool call logs estimated cost (zero for Feast / MLflow
  reads; non-zero for any call that triggers an LLM step).
- **Adding a tool** requires: Pydantic input + output models in
  `mcp_server.py`, a unit test calling the tool end-to-end without an
  external service (use `TestClient`), and a one-line entry in `README.md`'s
  "MCP tools" section for live-demo discoverability.

---

## 3. Research agent state contract

`src/auto_research/agents/research_graph.py` defines the LangGraph state
machine. The state object is the contract between nodes.

```python
class HypothesisType(str, Enum):
    FEATURE_EXTRACTION = "feature_extraction"      # high cost
    CONDITIONAL = "conditional"                    # low cost
    EVENT_WINDOW = "event_window"
    REGIME_CONDITIONAL = "regime_conditional"
    CROSS_SIGNAL_INTERACTION = "cross_signal_interaction"
    PURE_INFO_CONTENT = "pure_info_content"        # stops at T1

class ResearchState(BaseModel):
    session_id: str
    hypothesis: Hypothesis | None = None
    hypothesis_type: HypothesisType | None = None
    materialized_signal: SignalDefinition | None = None
    validation_tier: Literal["T1", "T2", "T3"] | None = None
    validation_report: InfoReport | BacktestReport | None = None
    decision: Literal["promote", "iterate", "kill"] | None = None
    critique: str | None = None             # LLM addendum, qualitative only
    memo_path: Path | None = None
    iteration_count: int = 0
    cost_usd_accumulated: float = 0.0       # circuit breaker input
```

Node interface: every node is `(state: ResearchState) → ResearchState`. State
is frozen (Pydantic). Modifications produce a new instance via
`state.model_copy(update={...})`.

`decision` is set **only** by the `decide` node, which reads
`validation_report` and applies `T1_GATE` / `T2_GATE` (see
`docs/BACKTEST.md`). The `critique` node runs in parallel and sets
`critique` — qualitative text appended to the memo, never read by `decide`.

---

## 4. Live critic contract

`src/auto_research/agents/live_critic.py` (Pydantic AI). Daily cron entry
point: `scripts/cron_daily_critic.sh`.

```python
class LiveCriticInput(BaseModel):
    as_of_date: date
    current_positions: list[PositionSnapshot]
    news_window_days: int = 1

class LiveCriticOutput(BaseModel):
    as_of_date: date
    per_position_haircut: dict[str, float]   # ticker → haircut ∈ [0, 1]
    flagged_overhangs: list[OverhangFlag]
    reasoning_per_position: dict[str, str]   # for memo trail
```

`per_position_haircut` is multiplicative on position sizing. Values < 1.0
reduce size. Value = 1.0 is no-op. **Value > 1.0 is invalid** (never an
upsize) — schema validator enforces.

Test path uses Pydantic AI's `TestModel` so unit tests run without an
Anthropic call.

---

## 5. Reliability primitives

`src/auto_research/agents/reliability.py`. Decorators / context managers
applied to every agent or worker that makes an LLM call.

```python
@circuit_breaker(failures=3)              # 3 consecutive failures → stop
@cost_cap(usd=5.00)                       # per-session hard limit
@max_iterations(n=10)                     # research-graph cap
@retry_with_backoff(max_retries=3)        # 5xx / rate limit
@fallback_model(primary="sonnet", fallback="haiku")
def my_llm_call(...): ...
```

Contract: every decorator is composable. Order matters — `cost_cap` outermost,
then `circuit_breaker`, then `retry_with_backoff`, then `fallback_model`. The
research graph applies all five via a single `@reliable_agent_node` composite.

---

## 6. Versioning & migration

- **Pydantic schemas (§1):** additive changes are non-breaking. Removals or
  renames bump the worker's `prompt_version` and require Feast FeatureView
  schema migration via `feast apply` + a one-off backfill script.
- **MCP tools (§2):** add freely. Renaming a tool breaks Claude Desktop /
  Cursor wiring — note in `README.md` if so.
- **Research state (§3):** additive. Removals require a checkpointer migration
  (the SqliteSaver stores serialized state).
- **Live critic (§4):** changing the `per_position_haircut` semantic
  (multiplicative → additive, etc.) is a breaking change to the paper
  portfolio engine — requires explicit user approval.
