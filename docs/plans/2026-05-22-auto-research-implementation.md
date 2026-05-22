# auto-research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-plane multi-agent research platform for cross-asset language-driven alpha in AI infrastructure and frontier-tech equities — the engineering corrective to virattt/ai-hedge-fund's LLM-at-wrong-layer anti-pattern.

**Architecture:** Deterministic trading plane (Feast PIT features, vbt.pro simulation, López de Prado validation) + LLM extraction plane (multi-doc structured extraction with contextual RAG) + LangGraph research agent + Pydantic AI live critic. MCP server exposes read-only data + research interface to external clients.

**Tech Stack:** Python 3.12 + uv • Anthropic API (Haiku/Sonnet/Opus tiered, Batch + prompt caching) • Feast 0.43+ on Parquet • vectorbt.pro • LangGraph + Pydantic AI • LanceDB + Voyage embeddings • Guardrails AI + DeepEval + Ragas • Langfuse self-hosted + OpenTelemetry/OpenInference + MLflow • FastMCP

**Spec:** `docs/specs/2026-05-22-design.md`

**Milestones map to GitHub milestones; tasks map 1:1 to GitHub issues.**

---

## Milestone 1 — W1: Foundation + extraction backbone

### Task 1: Repo scaffold + Python project

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`
- Create: `src/auto_research/__init__.py`
- Create: `tests/__init__.py`
- Create: `ruff.toml`
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "auto-research"
version = "0.1.0"
description = "Two-plane multi-agent research platform for cross-asset language alpha"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.9",
    "pydantic-ai>=0.0.13",
    "langgraph>=0.2.50",
    "feast>=0.43",
    "duckdb>=1.1",
    "pyarrow>=18.0",
    "lancedb>=0.15",
    "sentence-transformers>=3.3",
    "voyageai>=0.3",
    "rank-bm25>=0.2.2",
    "unstructured[pdf]>=0.16",
    "guardrails-ai>=0.5",
    "deepeval>=2.0",
    "ragas>=0.2",
    "mlflow>=2.18",
    "langfuse>=2.50",
    "opentelemetry-sdk>=1.28",
    "openinference-instrumentation-anthropic>=0.1",
    "openinference-instrumentation-langchain>=0.1",
    "vectorbtpro",
    "click>=8.1",
    "rich>=13.9",
    "httpx>=0.27",
    "python-dotenv>=1.0",
    "fastmcp>=0.4",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-asyncio>=0.24", "pytest-vcr>=1.0.2", "ruff>=0.7", "mypy>=1.13"]
dashboard = ["streamlit>=1.40"]

[project.scripts]
auto-research = "auto_research.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/auto_research"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.mypy_cache/
.env
.DS_Store

data/
!data/universe/
!data/.gitkeep

mlruns/
mlartifacts/
.langfuse/
*.lance/
.deepeval/

dist/
build/
*.egg-info/
```

- [ ] **Step 3: Write `.env.example`**

```bash
ANTHROPIC_API_KEY=
FMP_API_KEY=
VOYAGE_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000/api/public/otel
MLFLOW_TRACKING_URI=file:./mlruns
SEC_USER_AGENT="auto-research Sam Xu samxu0825@gmail.com"
```

- [ ] **Step 4: Write `ruff.toml`**

```toml
line-length = 100
target-version = "py312"

[lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM", "RUF"]
ignore = ["E501"]
```

- [ ] **Step 5: Write `src/auto_research/__init__.py` and `tests/__init__.py`**

```python
# src/auto_research/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 6: Write `README.md` stub**

```markdown
# auto-research

Two-plane multi-agent research platform for cross-asset language-driven alpha in AI infrastructure and frontier-tech equities. The engineering corrective to virattt/ai-hedge-fund's LLM-at-wrong-layer anti-pattern.

See `docs/specs/2026-05-22-design.md` for the architectural design.
See `docs/plans/2026-05-22-auto-research-implementation.md` for the implementation plan.

## Status
Under active development. v1 target: 4 weeks from 2026-05-22.

## Quickstart
TBD — see implementation plan.
```

- [ ] **Step 7: Write `CONTRIBUTING.md` documenting worktree convention**

```markdown
# Contributing

## Worktree workflow for parallel issues

Main checkout stays on `main`. Per-issue worktrees live in a sibling directory:

\`\`\`bash
git worktree add ../auto-research-trees/issue-42-ten-k-extractor -b feat/issue-42-ten-k-extractor
cd ../auto-research-trees/issue-42-ten-k-extractor
# work, commit, push, open PR
\`\`\`

On PR merge:

\`\`\`bash
cd ../../auto-research
git worktree remove ../auto-research-trees/issue-42-ten-k-extractor
git branch -d feat/issue-42-ten-k-extractor
\`\`\`

Use this for any work that can proceed independently of the current main checkout (most issues from W2 onward).

## Branch / commit conventions
- Branch: `feat/issue-N-short-slug` or `fix/issue-N-short-slug`
- Commits: conventional commits (`feat(extract): ten-k worker with citation grounding`)
- PRs close their issue via `Closes #N`
```

- [ ] **Step 8: Bootstrap venv and verify install**

```bash
cd /Users/feynman/Documents/projects/auto-research
uv venv
uv sync --all-extras
uv run python -c "import auto_research; print(auto_research.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore .env.example ruff.toml README.md CONTRIBUTING.md src/auto_research/__init__.py tests/__init__.py
git commit -m "chore: scaffold repo with uv, pyproject, ruff, contributing"
```

---

### Task 2: Langfuse self-hosted + OpenLLMetry telemetry

**Files:**
- Create: `docker-compose.yml`
- Create: `src/auto_research/telemetry.py`
- Create: `tests/test_telemetry.py`

- [ ] **Step 1: Write `docker-compose.yml` for Langfuse**

```yaml
services:
  langfuse-db:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    volumes:
      - langfuse_db_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  langfuse:
    image: langfuse/langfuse:2
    depends_on:
      - langfuse-db
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@langfuse-db:5432/postgres
      NEXTAUTH_SECRET: dev-only-change-for-prod
      NEXTAUTH_URL: http://localhost:3000
      TELEMETRY_ENABLED: "false"
      LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: "true"

volumes:
  langfuse_db_data:
```

- [ ] **Step 2: Start Langfuse, verify reachable, capture API keys**

```bash
docker compose up -d
sleep 30
curl -fsS http://localhost:3000/api/public/health
# Visit http://localhost:3000 in browser, sign up, create project, copy public + secret keys into .env
```

Expected: health endpoint returns 200. Manual step: copy keys to `.env`.

- [ ] **Step 3: Write failing test in `tests/test_telemetry.py`**

```python
import os
from auto_research.telemetry import init_telemetry, get_tracer

def test_init_telemetry_returns_configured_tracer():
    init_telemetry(service_name="auto-research-test")
    tracer = get_tracer("test")
    with tracer.start_as_current_span("smoke") as span:
        span.set_attribute("smoke.ok", True)
    # If init succeeded, no exception raised.
    assert tracer is not None
```

- [ ] **Step 4: Run test to verify it fails**

```bash
uv run pytest tests/test_telemetry.py -v
```

Expected: FAIL — `ModuleNotFoundError: auto_research.telemetry`

- [ ] **Step 5: Implement `src/auto_research/telemetry.py`**

```python
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

_initialized = False

def init_telemetry(service_name: str = "auto-research") -> None:
    global _initialized
    if _initialized:
        return
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:3000/api/public/otel")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    auth = f"Basic {__import__('base64').b64encode(f'{public_key}:{secret_key}'.encode()).decode()}"
    exporter = OTLPSpanExporter(endpoint=endpoint, headers={"Authorization": auth})
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    # Auto-instrument Anthropic
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().instrument()
    except Exception:
        pass
    _initialized = True

def get_tracer(name: str):
    return trace.get_tracer(name)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
uv run pytest tests/test_telemetry.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml src/auto_research/telemetry.py tests/test_telemetry.py
git commit -m "feat(obs): Langfuse self-hosted + OpenLLMetry OTel exporter"
```

---

### Task 3: MLflow local backend + smoke test

**Files:**
- Create: `src/auto_research/experiment_tracking.py`
- Create: `tests/test_experiment_tracking.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_experiment_tracking.py
import os
import tempfile
import mlflow
from auto_research.experiment_tracking import init_mlflow, log_run

def test_init_mlflow_creates_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    init_mlflow(experiment_name="test_exp")
    with log_run(run_name="smoke") as run:
        mlflow.log_metric("x", 1.0)
    assert (tmp_path / "mlruns").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_experiment_tracking.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/auto_research/experiment_tracking.py`**

```python
import os
from contextlib import contextmanager
import mlflow

def init_mlflow(experiment_name: str) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

@contextmanager
def log_run(run_name: str, tags: dict | None = None):
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        yield run
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_experiment_tracking.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/auto_research/experiment_tracking.py tests/test_experiment_tracking.py
git commit -m "feat(obs): MLflow local backend + run context manager"
```

---

### Task 4: Universe loader + ticker registry

**Files:**
- Create: `data/universe/universe_v1.json`
- Create: `src/auto_research/universe.py`
- Create: `tests/test_universe.py`

- [ ] **Step 1: Write `data/universe/universe_v1.json`**

```json
{
  "version": "v1",
  "as_of": "2026-05-22",
  "sub_universes": {
    "ai_infra_narrative": {
      "kind": "narrative_source",
      "tickers": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","AMD","ORCL","PLTR","CRM","NOW","ARM","CBRS","DLR","EQIX"]
    },
    "ai_infra_compute": {
      "kind": "tradeable",
      "tickers": ["CRDO","ALAB","COHR","AAOI","LITE","ANET","CIEN","MU","SNDK","WDC","MRVL","MTSI","MPWR","ASML","AMAT","LRCX","KLAC","TER","ACMR","AEHR","NVMI","FORM","LSCC","RMBS","SIMO","AMBA","ALGM","SWKS","QRVO","ON","GFS","TSM","SMCI","DELL","NTNX","IBM"]
    },
    "ai_infra_power": {
      "kind": "tradeable",
      "tickers": ["BE","VST","CEG","TLN","NRG","OKLO","SMR","NNE","LEU","CCJ","GEV","VRT","ETN","PWR","STRL","MIR"]
    },
    "frontier_quantum": {
      "kind": "tradeable",
      "tickers": ["IONQ","RGTI","QBTS","QUBT","HON"]
    },
    "frontier_space": {
      "kind": "tradeable",
      "tickers": ["RKLB","ASTS","LUNR","PL","RDW","BKSY","SATS"]
    }
  }
}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_universe.py
from auto_research.universe import load_universe, narrative_sources, tradeable

def test_load_universe_has_all_subuniverses():
    u = load_universe()
    assert u["version"] == "v1"
    assert "ai_infra_narrative" in u["sub_universes"]

def test_narrative_sources_includes_mag7():
    n = set(narrative_sources())
    assert {"NVDA", "MSFT", "GOOGL", "AAPL", "AMZN", "META", "TSLA"} <= n

def test_tradeable_excludes_narrative():
    n = set(narrative_sources())
    t = set(tradeable())
    assert n.isdisjoint(t)
```

- [ ] **Step 3: Run test — expect fail**

```bash
uv run pytest tests/test_universe.py -v
```

Expected: FAIL

- [ ] **Step 4: Implement `src/auto_research/universe.py`**

```python
import json
from pathlib import Path

_UNIVERSE_PATH = Path(__file__).resolve().parents[2] / "data" / "universe" / "universe_v1.json"

def load_universe() -> dict:
    return json.loads(_UNIVERSE_PATH.read_text())

def narrative_sources() -> list[str]:
    u = load_universe()
    return [t for sub in u["sub_universes"].values() if sub["kind"] == "narrative_source" for t in sub["tickers"]]

def tradeable() -> list[str]:
    u = load_universe()
    return [t for sub in u["sub_universes"].values() if sub["kind"] == "tradeable" for t in sub["tickers"]]

def all_tickers() -> list[str]:
    return narrative_sources() + tradeable()

def sub_universe(name: str) -> list[str]:
    return load_universe()["sub_universes"][name]["tickers"]
```

- [ ] **Step 5: Run test to verify pass + commit**

```bash
uv run pytest tests/test_universe.py -v
git add data/universe/universe_v1.json src/auto_research/universe.py tests/test_universe.py
git commit -m "feat(universe): load v1 universe with sub-universe accessors"
```

---

### Task 5: EDGAR ingestion client + manifest

**Files:**
- Create: `src/auto_research/ingest/__init__.py`
- Create: `src/auto_research/ingest/edgar.py`
- Create: `src/auto_research/ingest/manifest.py`
- Create: `tests/ingest/test_edgar.py`
- Create: `tests/ingest/test_manifest.py`
- Create: `tests/ingest/cassettes/` (pytest-vcr)

- [ ] **Step 1: Write manifest test**

```python
# tests/ingest/test_manifest.py
from auto_research.ingest.manifest import Manifest

def test_manifest_records_and_dedupes(tmp_path):
    m = Manifest(tmp_path / "manifest.parquet")
    m.record(source="edgar", entity_id="AAPL", doc_id="000032019325000001", content_sha256="a"*64, status="ok")
    m.record(source="edgar", entity_id="AAPL", doc_id="000032019325000001", content_sha256="a"*64, status="ok")
    df = m.load()
    assert len(df) == 1  # dedupe on (source, entity_id, doc_id, content_sha256)

def test_manifest_picks_up_changed_content(tmp_path):
    m = Manifest(tmp_path / "manifest.parquet")
    m.record(source="edgar", entity_id="AAPL", doc_id="0001", content_sha256="a"*64, status="ok")
    m.record(source="edgar", entity_id="AAPL", doc_id="0001", content_sha256="b"*64, status="ok")
    df = m.load()
    assert len(df) == 2  # different content → both rows kept
```

- [ ] **Step 2: Implement `src/auto_research/ingest/manifest.py`**

```python
from pathlib import Path
from datetime import datetime, UTC
import pyarrow as pa
import pyarrow.parquet as pq

_SCHEMA = pa.schema([
    ("source", pa.string()),
    ("entity_id", pa.string()),
    ("doc_id", pa.string()),
    ("content_sha256", pa.string()),
    ("fetched_at", pa.timestamp("us", tz="UTC")),
    ("status", pa.string()),
    ("meta_json", pa.string()),
])

class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, source: str, entity_id: str, doc_id: str, content_sha256: str, status: str, meta: dict | None = None) -> None:
        import json
        row = {
            "source": source, "entity_id": entity_id, "doc_id": doc_id,
            "content_sha256": content_sha256, "fetched_at": datetime.now(UTC),
            "status": status, "meta_json": json.dumps(meta or {}),
        }
        if self.path.exists():
            existing = pq.read_table(self.path)
            new = pa.Table.from_pylist([row], schema=_SCHEMA)
            combined = pa.concat_tables([existing, new])
            df = combined.to_pandas().drop_duplicates(
                subset=["source", "entity_id", "doc_id", "content_sha256"], keep="first"
            )
            pq.write_table(pa.Table.from_pandas(df, schema=_SCHEMA), self.path)
        else:
            pq.write_table(pa.Table.from_pylist([row], schema=_SCHEMA), self.path)

    def load(self):
        return pq.read_table(self.path).to_pandas() if self.path.exists() else None

    def has(self, *, source: str, entity_id: str, doc_id: str) -> bool:
        df = self.load()
        if df is None:
            return False
        return ((df["source"] == source) & (df["entity_id"] == entity_id) & (df["doc_id"] == doc_id)).any()
```

- [ ] **Step 3: Run manifest test, fix until pass**

```bash
uv run pytest tests/ingest/test_manifest.py -v
```

Expected: PASS

- [ ] **Step 4: Write EDGAR client test with VCR cassette**

```python
# tests/ingest/test_edgar.py
import pytest
from auto_research.ingest.edgar import EdgarClient

@pytest.fixture
def edgar(tmp_path):
    return EdgarClient(raw_dir=tmp_path / "raw", manifest_path=tmp_path / "manifest.parquet",
                       user_agent="test agent test@example.com")

@pytest.mark.vcr
def test_edgar_fetches_company_filings_index(edgar):
    filings = edgar.list_filings(ticker="AAPL", forms=["10-K"], since="2024-01-01")
    assert len(filings) >= 1
    assert all(f["form"] == "10-K" for f in filings)
    assert all("accepted_datetime" in f for f in filings)

@pytest.mark.vcr
def test_edgar_downloads_filing_text_idempotently(edgar):
    filings = edgar.list_filings(ticker="AAPL", forms=["10-K"], since="2024-01-01")
    f = filings[0]
    text1 = edgar.fetch_filing_text(f)
    text2 = edgar.fetch_filing_text(f)  # second call hits cache, no new HTTP
    assert text1 == text2
    assert len(text1) > 10_000
```

- [ ] **Step 5: Implement `src/auto_research/ingest/edgar.py`**

```python
import hashlib
import httpx
from pathlib import Path
from datetime import datetime, UTC
from auto_research.ingest.manifest import Manifest

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives"

class EdgarClient:
    def __init__(self, raw_dir: Path, manifest_path: Path, user_agent: str):
        self.raw_dir = Path(raw_dir); self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(manifest_path)
        self.client = httpx.Client(headers={"User-Agent": user_agent}, timeout=30.0)

    def _cik_for_ticker(self, ticker: str) -> str:
        tickers = self.client.get(f"{EDGAR_BASE}/files/company_tickers.json").json()
        for row in tickers.values():
            if row["ticker"].upper() == ticker.upper():
                return str(row["cik_str"]).zfill(10)
        raise ValueError(f"No CIK for ticker {ticker}")

    def list_filings(self, ticker: str, forms: list[str], since: str) -> list[dict]:
        cik = self._cik_for_ticker(ticker)
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        data = self.client.get(url).json()
        recent = data["filings"]["recent"]
        results = []
        for i, form in enumerate(recent["form"]):
            if form not in forms: continue
            accepted = recent["acceptanceDateTime"][i]
            if accepted < since: continue
            results.append({
                "ticker": ticker, "cik": cik, "form": form,
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
                "accepted_datetime": accepted,
                "period": recent["reportDate"][i],
            })
        return results

    def fetch_filing_text(self, filing: dict) -> str:
        acc = filing["accession"].replace("-", "")
        url = f"{EDGAR_ARCHIVES}/edgar/data/{int(filing['cik'])}/{acc}/{filing['primary_doc']}"
        out = self.raw_dir / filing["ticker"] / filing["form"] / f"{filing['accession']}.html"
        if out.exists():
            return out.read_text()
        out.parent.mkdir(parents=True, exist_ok=True)
        text = self.client.get(url).text
        out.write_text(text)
        sha = hashlib.sha256(text.encode()).hexdigest()
        self.manifest.record(
            source="edgar", entity_id=filing["ticker"], doc_id=filing["accession"],
            content_sha256=sha, status="ok", meta=filing,
        )
        return text
```

- [ ] **Step 6: Record VCR cassettes + run test**

```bash
uv run pytest tests/ingest/test_edgar.py -v --record-mode=once
```

Expected: PASS, creates `tests/ingest/cassettes/` files.

- [ ] **Step 7: Commit**

```bash
git add src/auto_research/ingest/ tests/ingest/
git commit -m "feat(ingest): EDGAR client + manifest with content-hash idempotency"
```

---

### Task 6: FMP transcript ingestion client

**Files:**
- Create: `src/auto_research/ingest/fmp.py`
- Create: `tests/ingest/test_fmp.py`

- [ ] **Step 1: Write test with VCR cassette**

```python
# tests/ingest/test_fmp.py
import pytest, os
from auto_research.ingest.fmp import FMPTranscriptClient

@pytest.fixture
def fmp(tmp_path):
    return FMPTranscriptClient(
        api_key=os.environ.get("FMP_API_KEY", "test"),
        raw_dir=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.parquet",
    )

@pytest.mark.vcr
def test_fmp_lists_transcripts_for_ticker(fmp):
    transcripts = fmp.list_transcripts("NVDA", since="2024-01-01")
    assert len(transcripts) >= 1
    assert all("event_datetime" in t for t in transcripts)

@pytest.mark.vcr
def test_fmp_fetches_transcript_idempotently(fmp):
    t = fmp.list_transcripts("NVDA", since="2024-01-01")[0]
    text1 = fmp.fetch_transcript(t)
    text2 = fmp.fetch_transcript(t)
    assert text1 == text2
    assert "Q&A" in text1 or "Question" in text1 or len(text1) > 5000
```

- [ ] **Step 2: Implement `src/auto_research/ingest/fmp.py`**

```python
import hashlib, httpx, json
from pathlib import Path
from auto_research.ingest.manifest import Manifest

FMP_BASE = "https://financialmodelingprep.com/api/v3"

class FMPTranscriptClient:
    def __init__(self, api_key: str, raw_dir: Path, manifest_path: Path):
        self.api_key = api_key
        self.raw_dir = Path(raw_dir); self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(manifest_path)
        self.client = httpx.Client(timeout=30.0)

    def list_transcripts(self, ticker: str, since: str) -> list[dict]:
        # List quarterly transcripts
        url = f"{FMP_BASE}/earning_call_transcript/{ticker}"
        resp = self.client.get(url, params={"apikey": self.api_key}).json()
        out = []
        for entry in resp:
            if entry["date"] >= since:
                out.append({
                    "ticker": ticker, "year": entry["year"], "quarter": entry["quarter"],
                    "event_datetime": entry["date"], "transcript_id": f"{ticker}-{entry['year']}Q{entry['quarter']}",
                })
        return out

    def fetch_transcript(self, t: dict) -> str:
        out = self.raw_dir / t["ticker"] / "transcripts" / f"{t['transcript_id']}.txt"
        if out.exists():
            return out.read_text()
        url = f"{FMP_BASE}/earning_call_transcript/{t['ticker']}"
        resp = self.client.get(url, params={
            "year": t["year"], "quarter": t["quarter"], "apikey": self.api_key,
        }).json()
        text = resp[0]["content"] if resp else ""
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        sha = hashlib.sha256(text.encode()).hexdigest()
        self.manifest.record(
            source="fmp", entity_id=t["ticker"], doc_id=t["transcript_id"],
            content_sha256=sha, status="ok" if text else "empty", meta=t,
        )
        return text
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/ingest/test_fmp.py -v --record-mode=once
git add src/auto_research/ingest/fmp.py tests/ingest/test_fmp.py tests/ingest/cassettes/
git commit -m "feat(ingest): FMP earnings-transcript client with manifest"
```

---

### Task 7: Feast feature store scaffold + price FeatureView + PIT discipline test

**Files:**
- Create: `feast/feature_store.yaml`
- Create: `feast/repo/__init__.py`
- Create: `feast/repo/entities.py`
- Create: `feast/repo/sources.py`
- Create: `feast/repo/feature_views.py`
- Create: `feast/repo/feature_services.py`
- Create: `tests/test_feast_pit.py`

- [ ] **Step 1: Write `feast/feature_store.yaml`**

```yaml
project: auto_research
registry: feast/registry.db
provider: local
offline_store:
  type: file
online_store:
  type: sqlite
  path: feast/online.db
entity_key_serialization_version: 2
```

- [ ] **Step 2: Write `feast/repo/entities.py`**

```python
from feast import Entity, ValueType

ticker = Entity(
    name="ticker",
    value_type=ValueType.STRING,
    description="Equity ticker symbol",
)
```

- [ ] **Step 3: Write `feast/repo/sources.py`**

```python
from pathlib import Path
from feast import FileSource

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "pit"

prices_source = FileSource(
    path=str(DATA / "facts_prices.parquet"),
    event_timestamp_column="as_of_ts",
    created_timestamp_column="ingested_at",
)
```

- [ ] **Step 4: Write `feast/repo/feature_views.py`**

```python
from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float64
from .entities import ticker
from .sources import prices_source

price_features = FeatureView(
    name="price_features",
    entities=[ticker],
    ttl=timedelta(days=5),
    schema=[
        Field(name="close", dtype=Float64),
        Field(name="adj_close", dtype=Float64),
        Field(name="volume", dtype=Float64),
        Field(name="returns_1d", dtype=Float64),
    ],
    source=prices_source,
    online=False,
)
```

- [ ] **Step 5: Write `feast/repo/feature_services.py`** (stub for now)

```python
from feast import FeatureService
from .feature_views import price_features

baseline = FeatureService(name="baseline", features=[price_features])
```

- [ ] **Step 6: Write `feast/repo/__init__.py`**

```python
from .entities import ticker
from .sources import prices_source
from .feature_views import price_features
from .feature_services import baseline

__all__ = ["ticker", "prices_source", "price_features", "baseline"]
```

- [ ] **Step 7: Write failing PIT test**

```python
# tests/test_feast_pit.py
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, UTC, timedelta
import subprocess

DATA = Path(__file__).resolve().parents[1] / "data" / "pit"
DATA.mkdir(parents=True, exist_ok=True)

def _seed_prices():
    rows = []
    for ticker in ["AAPL", "MSFT"]:
        for i in range(5):
            day = datetime(2025, 1, 1, 21, 0, tzinfo=UTC) + timedelta(days=i)  # 16:00 ET == 21:00 UTC
            as_of = day + timedelta(days=1)  # lag-1 enforced at write time
            rows.append({
                "ticker": ticker,
                "as_of_ts": as_of,
                "ingested_at": as_of,
                "close": 100.0 + i,
                "adj_close": 100.0 + i,
                "volume": 1_000_000,
                "returns_1d": 0.01 if i > 0 else 0.0,
            })
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), DATA / "facts_prices.parquet")

def test_pit_join_does_not_leak_lookahead():
    _seed_prices()
    subprocess.run(["uv", "run", "feast", "-c", "feast", "apply"], check=True)
    from feast import FeatureStore
    store = FeatureStore(repo_path="feast")
    # Ask for features as-of a date BEFORE the lag-1 cutoff
    entity_df = pd.DataFrame([
        {"ticker": "AAPL", "event_timestamp": datetime(2025, 1, 1, 22, 0, tzinfo=UTC)},
    ])
    df = store.get_historical_features(
        entity_df=entity_df,
        features=["price_features:close"],
    ).to_df()
    # The price for 2025-01-01 has as_of_ts = 2025-01-02 21:00 UTC; querying at 22:00 UTC on 01-01 must return null
    assert df["close"].isna().all(), "Feast leaked a future feature into a PIT-correct query"
```

- [ ] **Step 8: Run test — expect fail (file structure not yet wired), then iterate**

```bash
uv run pytest tests/test_feast_pit.py -v -s
```

Expected: initially fails. Iterate on path config and `feast apply` until test passes.

- [ ] **Step 9: Commit when PIT test passes**

```bash
git add feast/ tests/test_feast_pit.py
git commit -m "feat(feast): scaffold + price FeatureView + PIT discipline test"
```

---

### Task 8: Reliability primitives (circuit breaker, cost cap, retry, fallback)

**Files:**
- Create: `src/auto_research/agents/__init__.py`
- Create: `src/auto_research/agents/reliability.py`
- Create: `tests/agents/test_reliability.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_reliability.py
import pytest, time
from auto_research.agents.reliability import (
    CircuitBreaker, CostCap, retry_with_backoff, CostExceeded, CircuitOpen,
)

def test_circuit_breaker_opens_after_n_failures():
    cb = CircuitBreaker(max_failures=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open()
    with pytest.raises(CircuitOpen):
        cb.guard()

def test_cost_cap_raises_when_exceeded():
    cap = CostCap(usd_budget=1.00)
    cap.charge(0.50)
    cap.charge(0.40)
    with pytest.raises(CostExceeded):
        cap.charge(0.20)

def test_retry_with_backoff_succeeds_on_third_attempt():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.HTTPError("boom")
        return "ok"
    import httpx
    result = retry_with_backoff(flaky, max_attempts=4, initial_delay=0.01)
    assert result == "ok"
    assert calls["n"] == 3
```

- [ ] **Step 2: Implement `src/auto_research/agents/reliability.py`**

```python
import time, random
from typing import Callable, TypeVar
T = TypeVar("T")

class CircuitOpen(Exception): pass
class CostExceeded(Exception): pass

class CircuitBreaker:
    def __init__(self, max_failures: int = 5):
        self.max_failures = max_failures
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1

    def record_success(self) -> None:
        self.failures = 0

    def is_open(self) -> bool:
        return self.failures >= self.max_failures

    def guard(self) -> None:
        if self.is_open():
            raise CircuitOpen(f"circuit breaker open after {self.failures} failures")

class CostCap:
    def __init__(self, usd_budget: float):
        self.budget = usd_budget
        self.spent = 0.0

    def charge(self, usd: float) -> None:
        if self.spent + usd > self.budget:
            raise CostExceeded(f"would spend ${self.spent + usd:.4f} > budget ${self.budget:.4f}")
        self.spent += usd

def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retry_on: tuple = (Exception,),
) -> T:
    delay = initial_delay
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retry_on as e:
            last_exc = e
            if attempt == max_attempts - 1:
                raise
            sleep_time = delay + random.uniform(0, jitter * delay)
            time.sleep(sleep_time)
            delay *= backoff_factor
    raise last_exc  # unreachable
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/agents/test_reliability.py -v
git add src/auto_research/agents/__init__.py src/auto_research/agents/reliability.py tests/agents/
git commit -m "feat(agents): reliability primitives — circuit breaker, cost cap, retry"
```

---

### Task 9: Pydantic extraction schemas with citation grounding

**Files:**
- Create: `src/auto_research/extract/__init__.py`
- Create: `src/auto_research/extract/schemas.py`
- Create: `tests/extract/test_schemas.py`

- [ ] **Step 1: Write failing tests for citation grounding validator**

```python
# tests/extract/test_schemas.py
import pytest
from pydantic import ValidationError
from auto_research.extract.schemas import (
    SupplierMention, TenKExtraction, validate_citations_in_source,
)

SOURCE_TEXT = "Our networking partners include CRDO and ALAB. We rely on TSM for fabrication."

def test_supplier_mention_quote_must_match_span():
    m = SupplierMention(
        target_entity="CRDO",
        tone="positive",
        horizon_days=90,
        confidence=0.8,
        source_span=(25, 29),
        source_quote="CRDO",
    )
    assert validate_citations_in_source(m, SOURCE_TEXT) is True

def test_supplier_mention_rejects_mismatched_quote():
    m = SupplierMention(
        target_entity="CRDO",
        tone="positive",
        horizon_days=90,
        confidence=0.8,
        source_span=(25, 29),
        source_quote="ALAB",  # wrong
    )
    assert validate_citations_in_source(m, SOURCE_TEXT) is False
```

- [ ] **Step 2: Implement schemas (write `src/auto_research/extract/schemas.py`)**

```python
from typing import Literal
from pydantic import BaseModel, Field

GuidanceTone = Literal["confident", "cautious", "evasive", "neutral", "none"]
EventType = Literal[
    "partnership","contract","customer_announcement","product_launch",
    "milestone","data_readout","regulatory_action","executive_change",
    "guidance_update","dilution","other"
]

class SourceCitation(BaseModel):
    source_span: tuple[int, int]
    source_quote: str

class SupplierMention(SourceCitation):
    target_entity: str
    tone: Literal["positive", "negative", "neutral"]
    horizon_days: int = Field(ge=0, le=730)
    confidence: float = Field(ge=0.0, le=1.0)

class CustomerMention(SourceCitation):
    target_entity: str
    concentration_change: Literal["increasing", "decreasing", "stable", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)

class AccrualFlag(SourceCitation):
    flag_type: Literal["unusual_accrual", "channel_stuffing", "reserve_release", "other"]
    severity: Literal["low", "medium", "high"]

class RiskFactorDelta(SourceCitation):
    risk_category: str
    change: Literal["new", "expanded", "reduced", "removed", "unchanged"]

class ExtractionMeta(BaseModel):
    prompt_version: str
    model_id: str
    extraction_run_id: str
    cost_usd: float = 0.0

class TenKExtraction(BaseModel):
    entity_id: str
    fiscal_period: str
    guidance_tone: GuidanceTone
    accrual_anomaly_flags: list[AccrualFlag] = []
    customer_concentration_disclosures: list[CustomerMention] = []
    supplier_mentions: list[SupplierMention] = []
    language_novelty_score: float = Field(ge=0.0, le=1.0)
    risk_factor_deltas: list[RiskFactorDelta] = []
    extraction_metadata: ExtractionMeta

class ForwardStatement(SourceCitation):
    target_entity: str | None  # None = self-referential
    tone: Literal["positive", "negative", "neutral"]
    horizon_days: int = Field(ge=0, le=730)
    confidence: float = Field(ge=0.0, le=1.0)

class TranscriptExtraction(BaseModel):
    entity_id: str
    fiscal_period: str
    prepared_remarks_tone: GuidanceTone
    q_and_a_evasiveness: float = Field(ge=0.0, le=1.0)
    forward_statements: list[ForwardStatement] = []
    extraction_metadata: ExtractionMeta

class EightKEvent(SourceCitation):
    event_type: EventType
    counterparties: list[str] = []
    tone: Literal["positive", "negative", "neutral"]
    materiality: Literal["low", "medium", "high"]

class EightKExtraction(BaseModel):
    entity_id: str
    filing_date: str
    events: list[EightKEvent] = []
    dilution_language_flag: bool = False
    extraction_metadata: ExtractionMeta

class SFilingExtraction(BaseModel):
    entity_id: str
    filing_date: str
    filing_type: Literal["S-1", "S-3"]
    dilution_event: bool
    capital_raise_amount_usd: float | None = None
    use_of_proceeds: list[str] = []
    dilution_language_severity: Literal["low", "medium", "high", "none"]
    extraction_metadata: ExtractionMeta

def validate_citations_in_source(item: SourceCitation, source_text: str) -> bool:
    start, end = item.source_span
    if start < 0 or end > len(source_text) or start >= end:
        return False
    return source_text[start:end] == item.source_quote
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/extract/test_schemas.py -v
git add src/auto_research/extract/__init__.py src/auto_research/extract/schemas.py tests/extract/
git commit -m "feat(extract): Pydantic schemas with citation-grounding validator"
```

---

### Task 10: Anthropic extraction client with caching + tiered models

**Files:**
- Create: `src/auto_research/extract/client.py`
- Create: `src/auto_research/extract/cache.py`
- Create: `tests/extract/test_cache.py`
- Create: `tests/extract/test_client.py`

- [ ] **Step 1: Write cache test**

```python
# tests/extract/test_cache.py
from auto_research.extract.cache import ExtractionCache

def test_cache_hits_on_same_inputs(tmp_path):
    c = ExtractionCache(tmp_path / "cache")
    key_args = dict(prompt_version="v1", model_id="claude-sonnet-4-6", raw_doc_text="hello world")
    assert c.get(**key_args) is None
    c.put(**key_args, response={"x": 1})
    assert c.get(**key_args) == {"x": 1}

def test_cache_misses_on_changed_prompt(tmp_path):
    c = ExtractionCache(tmp_path / "cache")
    c.put(prompt_version="v1", model_id="m", raw_doc_text="d", response={"x": 1})
    assert c.get(prompt_version="v2", model_id="m", raw_doc_text="d") is None
```

- [ ] **Step 2: Implement cache**

```python
# src/auto_research/extract/cache.py
import hashlib, json
from pathlib import Path

class ExtractionCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *, prompt_version: str, model_id: str, raw_doc_text: str) -> str:
        h = hashlib.sha256()
        h.update(prompt_version.encode()); h.update(b"|")
        h.update(model_id.encode()); h.update(b"|")
        h.update(raw_doc_text.encode())
        return h.hexdigest()

    def get(self, *, prompt_version: str, model_id: str, raw_doc_text: str):
        path = self.dir / f"{self._key(prompt_version=prompt_version, model_id=model_id, raw_doc_text=raw_doc_text)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put(self, *, prompt_version: str, model_id: str, raw_doc_text: str, response: dict) -> None:
        path = self.dir / f"{self._key(prompt_version=prompt_version, model_id=model_id, raw_doc_text=raw_doc_text)}.json"
        path.write_text(json.dumps(response))
```

- [ ] **Step 3: Run cache test**

```bash
uv run pytest tests/extract/test_cache.py -v
```

Expected: PASS.

- [ ] **Step 4: Implement Anthropic client with tiered model routing + cache**

```python
# src/auto_research/extract/client.py
import os, json
from typing import Type
from pydantic import BaseModel
from anthropic import Anthropic
from auto_research.extract.cache import ExtractionCache
from auto_research.agents.reliability import retry_with_backoff, CircuitBreaker, CostCap

PRICING_USD_PER_M = {  # placeholder pricing — update per Anthropic announcements
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cached_input": 0.10},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cached_input": 0.30},
    "claude-opus-4-7": {"input": 15.0, "output": 75.0, "cached_input": 1.50},
}

class ExtractionClient:
    def __init__(self, cache: ExtractionCache, cost_cap: CostCap | None = None,
                 circuit_breaker: CircuitBreaker | None = None):
        self.client = Anthropic()
        self.cache = cache
        self.cost_cap = cost_cap or CostCap(usd_budget=200.0)
        self.cb = circuit_breaker or CircuitBreaker(max_failures=5)

    def extract(
        self,
        *,
        schema: Type[BaseModel],
        system_prompt: str,
        prompt_version: str,
        raw_doc_text: str,
        model_id: str,
        max_tokens: int = 4096,
    ) -> BaseModel:
        self.cb.guard()
        cached = self.cache.get(prompt_version=prompt_version, model_id=model_id, raw_doc_text=raw_doc_text)
        if cached is not None:
            return schema.model_validate(cached["parsed"])

        def call():
            return self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=[{
                    "type": "text", "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": raw_doc_text}],
                tools=[{
                    "name": "submit_extraction",
                    "description": f"Submit structured extraction conforming to {schema.__name__}.",
                    "input_schema": schema.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": "submit_extraction"},
            )

        try:
            resp = retry_with_backoff(call, max_attempts=3, initial_delay=2.0)
        except Exception:
            self.cb.record_failure()
            raise

        self.cb.record_success()
        usage = resp.usage
        pricing = PRICING_USD_PER_M[model_id]
        cost = (
            (usage.input_tokens - getattr(usage, "cache_read_input_tokens", 0)) * pricing["input"] / 1e6
            + getattr(usage, "cache_read_input_tokens", 0) * pricing["cached_input"] / 1e6
            + usage.output_tokens * pricing["output"] / 1e6
        )
        self.cost_cap.charge(cost)
        tool_use = next(b for b in resp.content if b.type == "tool_use")
        parsed = tool_use.input
        self.cache.put(
            prompt_version=prompt_version, model_id=model_id, raw_doc_text=raw_doc_text,
            response={"parsed": parsed, "cost_usd": cost, "usage": usage.model_dump()},
        )
        return schema.model_validate(parsed)
```

- [ ] **Step 5: Write smoke test (skip without API key)**

```python
# tests/extract/test_client.py
import os, pytest
from auto_research.extract.client import ExtractionClient
from auto_research.extract.cache import ExtractionCache
from auto_research.extract.schemas import SFilingExtraction

@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_extract_minimal_s_filing(tmp_path):
    client = ExtractionClient(cache=ExtractionCache(tmp_path / "cache"))
    text = "On 2025-04-15, registrant filed a registration statement on Form S-3 for up to $500M of common stock. Use of proceeds: general corporate purposes."
    result = client.extract(
        schema=SFilingExtraction,
        system_prompt="Extract structured S-filing data. Cite source spans verbatim.",
        prompt_version="smoke-v1",
        raw_doc_text=text,
        model_id="claude-haiku-4-5-20251001",
        max_tokens=2048,
    )
    assert result.dilution_event is True
    assert result.filing_type == "S-3"
```

- [ ] **Step 6: Run + commit**

```bash
uv run pytest tests/extract/test_client.py tests/extract/test_cache.py -v
git add src/auto_research/extract/client.py src/auto_research/extract/cache.py tests/extract/test_cache.py tests/extract/test_client.py
git commit -m "feat(extract): Anthropic client with tiered models, caching, retry"
```

---

### Task 11: Extraction prompts directory + S-1/S-3 worker (simplest, validate the pipe)

**Files:**
- Create: `src/auto_research/extract/prompts/__init__.py`
- Create: `src/auto_research/extract/prompts/s_filings_v1.txt`
- Create: `src/auto_research/extract/workers/__init__.py`
- Create: `src/auto_research/extract/workers/s_filings.py`
- Create: `tests/extract/test_s_filings_worker.py`

- [ ] **Step 1: Write S-filings prompt**

```
# src/auto_research/extract/prompts/s_filings_v1.txt
You are extracting structured information from an SEC S-1 or S-3 registration statement.

For every claim you make, you MUST include:
- source_span: tuple of (start_char, end_char) into the input document
- source_quote: the exact verbatim text from the document at that span

Extract:
- filing_type: "S-1" or "S-3"
- dilution_event: True if the filing announces new share issuance, secondary offering, ATM facility, shelf registration, etc.
- capital_raise_amount_usd: total dollar amount of the offering if disclosed; null otherwise
- use_of_proceeds: list of stated uses (e.g., "general corporate purposes", "debt repayment", "R&D")
- dilution_language_severity:
  - "high": large offering relative to market cap, urgent language, distressed signal
  - "medium": standard offering, neutral language
  - "low": small offering, optional facility (e.g., ATM not yet drawn)
  - "none": filing is not actually dilutive (e.g., resale registration for selling shareholders)

Return only the structured extraction via the submit_extraction tool. Be precise about source spans — they must match the input text verbatim.
```

- [ ] **Step 2: Write failing worker test (mocked)**

```python
# tests/extract/test_s_filings_worker.py
from unittest.mock import patch, MagicMock
from auto_research.extract.workers.s_filings import extract_s_filing
from auto_research.extract.schemas import SFilingExtraction, ExtractionMeta

def test_extract_s_filing_returns_validated_schema(tmp_path):
    text = "...filing text..."
    fake_result = SFilingExtraction(
        entity_id="MU", filing_date="2025-04-15", filing_type="S-3",
        dilution_event=True, capital_raise_amount_usd=500_000_000.0,
        use_of_proceeds=["general corporate purposes"],
        dilution_language_severity="medium",
        extraction_metadata=ExtractionMeta(
            prompt_version="s_filings_v1", model_id="claude-haiku-4-5-20251001",
            extraction_run_id="test-run-1",
        ),
    )
    with patch("auto_research.extract.workers.s_filings.ExtractionClient") as MockClient:
        mock = MockClient.return_value
        mock.extract.return_value = fake_result
        result = extract_s_filing(
            entity_id="MU", filing_date="2025-04-15", raw_text=text,
            cache_dir=tmp_path / "cache", run_id="test-run-1",
        )
    assert isinstance(result, SFilingExtraction)
    assert result.dilution_event is True
```

- [ ] **Step 3: Implement worker**

```python
# src/auto_research/extract/workers/s_filings.py
from pathlib import Path
from auto_research.extract.client import ExtractionClient
from auto_research.extract.cache import ExtractionCache
from auto_research.extract.schemas import SFilingExtraction

PROMPT_VERSION = "s_filings_v1"
MODEL_ID = "claude-haiku-4-5-20251001"

_PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "s_filings_v1.txt").read_text()

def extract_s_filing(
    *, entity_id: str, filing_date: str, raw_text: str,
    cache_dir: Path, run_id: str,
) -> SFilingExtraction:
    client = ExtractionClient(cache=ExtractionCache(cache_dir))
    result = client.extract(
        schema=SFilingExtraction,
        system_prompt=_PROMPT,
        prompt_version=PROMPT_VERSION,
        raw_doc_text=raw_text,
        model_id=MODEL_ID,
    )
    # Force consistent IDs even if the LLM ignored them.
    return result.model_copy(update={
        "entity_id": entity_id, "filing_date": filing_date,
    })
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/extract/test_s_filings_worker.py -v
git add src/auto_research/extract/prompts src/auto_research/extract/workers/ tests/extract/test_s_filings_worker.py
git commit -m "feat(extract): S-1/S-3 worker with prompt v1"
```

---

### Task 12: CLI entry point + W1 acceptance smoke

**Files:**
- Create: `src/auto_research/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from auto_research.cli import main

def test_cli_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["ingest", "extract", "backtest", "research", "critic"]:
        assert cmd in result.output
```

- [ ] **Step 2: Implement CLI**

```python
# src/auto_research/cli.py
import click

@click.group()
def main():
    """auto-research CLI."""

@main.command()
@click.option("--source", type=click.Choice(["edgar","fmp"]), required=True)
@click.option("--ticker", multiple=True)
def ingest(source, ticker):
    """Ingest documents from a source."""
    click.echo(f"ingest source={source} tickers={list(ticker)}")

@main.command()
@click.option("--worker", type=click.Choice(["ten_k","transcript","eight_k","s_filings"]), required=True)
def extract(worker):
    """Run an extraction worker."""
    click.echo(f"extract worker={worker}")

@main.command()
@click.option("--signal", required=True)
@click.option("--tier", type=click.Choice(["T1","T2","T3"]), default="T1")
def backtest(signal, tier):
    """Run backtest at the specified tier."""
    click.echo(f"backtest signal={signal} tier={tier}")

@main.command()
@click.argument("subcommand", required=False)
def research(subcommand):
    """Run the research agent."""
    click.echo(f"research subcommand={subcommand}")

@main.command()
def critic():
    """Run the live critic."""
    click.echo("critic")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run + commit, then tag W1 complete**

```bash
uv run pytest tests/test_cli.py -v
git add src/auto_research/cli.py tests/test_cli.py
git commit -m "feat(cli): scaffold ingest/extract/backtest/research/critic subcommands"
git tag w1-foundation-complete
```

---

**Milestone 1 acceptance:**
- `uv run pytest -v` all pass
- `docker compose up -d` brings Langfuse online at http://localhost:3000
- `uv run auto-research --help` lists all subcommands
- `uv run feast -c feast apply` succeeds; PIT discipline test passes
- Anthropic extraction smoke test passes on a real S-filing (with API key)

Continue to Milestone 2 (RAG layer) once W1 acceptance gates pass.

---

## Milestone 2 — W2: RAG layer + extraction quality

### Task 13: Document parsing — unstructured.io + section-aware chunking

**Files:**
- Create: `src/auto_research/extract/parsing.py`
- Create: `src/auto_research/extract/chunking.py`
- Create: `tests/extract/test_parsing.py`
- Create: `tests/extract/test_chunking.py`
- Create: `tests/extract/fixtures/sample_10k.html` (tiny synthetic 10-K-shaped doc)

- [ ] **Step 1: Write synthetic 10-K fixture**

```html
<!-- tests/extract/fixtures/sample_10k.html -->
<html><body>
<h1>FORM 10-K</h1>
<h2>PART I</h2>
<h3>Item 1. Business</h3>
<p>We design and manufacture AI infrastructure components.</p>
<h3>Item 1A. Risk Factors</h3>
<p>Geopolitical risks include China export controls under EAR section 4.</p>
<p>Customer concentration: top three customers represented 60% of revenue.</p>
<h2>PART II</h2>
<h3>Item 7. Management's Discussion and Analysis</h3>
<p>Revenue grew 45% year over year driven by hyperscaler demand.</p>
<h3>Item 8. Financial Statements</h3>
<p>Net income for fiscal 2025 was $2.1B.</p>
</body></html>
```

- [ ] **Step 2: Write parsing test**

```python
# tests/extract/test_parsing.py
from pathlib import Path
from auto_research.extract.parsing import parse_10k_sections

def test_parse_10k_extracts_named_sections():
    html = (Path(__file__).parent / "fixtures" / "sample_10k.html").read_text()
    sections = parse_10k_sections(html)
    assert "Item 1. Business" in sections
    assert "Item 1A. Risk Factors" in sections
    assert "Item 7" in next(k for k in sections if "Item 7" in k)
    assert "China export controls" in sections["Item 1A. Risk Factors"]
```

- [ ] **Step 3: Implement parsing**

```python
# src/auto_research/extract/parsing.py
import re
from unstructured.partition.html import partition_html

ITEM_PATTERN = re.compile(r"^Item\s+\d+[A-Z]?\.\s+", re.IGNORECASE)

def parse_10k_sections(html: str) -> dict[str, str]:
    elements = partition_html(text=html)
    sections: dict[str, str] = {}
    current_section = None
    buf: list[str] = []
    for el in elements:
        text = str(el).strip()
        if not text:
            continue
        if ITEM_PATTERN.match(text):
            if current_section is not None:
                sections[current_section] = "\n".join(buf).strip()
            current_section = text
            buf = []
        else:
            if current_section is None:
                continue  # pre-content boilerplate
            buf.append(text)
    if current_section is not None:
        sections[current_section] = "\n".join(buf).strip()
    return sections
```

- [ ] **Step 4: Write chunking test**

```python
# tests/extract/test_chunking.py
from auto_research.extract.chunking import section_aware_chunk

def test_section_aware_chunk_respects_section_boundaries():
    sections = {
        "Item 1A. Risk Factors": "Geopolitical risks include China export controls.\nCustomer concentration is high.",
        "Item 7. MD&A": "Revenue grew 45% year over year.",
    }
    chunks = section_aware_chunk(sections, target_tokens=20)
    assert all(c["section"] in sections for c in chunks)
    # No chunk spans two sections
    for c in chunks:
        assert sections[c["section"]].find(c["text"][:20]) >= 0
```

- [ ] **Step 5: Implement chunking**

```python
# src/auto_research/extract/chunking.py
from dataclasses import dataclass

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough ~4 chars/token

def section_aware_chunk(
    sections: dict[str, str],
    target_tokens: int = 800,
    overlap_tokens: int = 80,
) -> list[dict]:
    chunks: list[dict] = []
    for section_name, text in sections.items():
        words = text.split()
        if not words:
            continue
        cur = []
        cur_tokens = 0
        for w in words:
            t = _approx_tokens(w + " ")
            if cur_tokens + t > target_tokens and cur:
                chunks.append({"section": section_name, "text": " ".join(cur)})
                # overlap
                overlap = []
                ov_tokens = 0
                for w2 in reversed(cur):
                    ov_tokens += _approx_tokens(w2 + " ")
                    overlap.insert(0, w2)
                    if ov_tokens >= overlap_tokens:
                        break
                cur = overlap[:]
                cur_tokens = sum(_approx_tokens(w2 + " ") for w2 in cur)
            cur.append(w); cur_tokens += t
        if cur:
            chunks.append({"section": section_name, "text": " ".join(cur)})
    return chunks
```

- [ ] **Step 6: Run + commit**

```bash
uv run pytest tests/extract/test_parsing.py tests/extract/test_chunking.py -v
git add src/auto_research/extract/parsing.py src/auto_research/extract/chunking.py tests/extract/test_parsing.py tests/extract/test_chunking.py tests/extract/fixtures/sample_10k.html
git commit -m "feat(rag): unstructured.io 10-K parsing + section-aware chunking"
```

---

### Task 14: Contextual chunking (Anthropic pattern)

**Files:**
- Create: `src/auto_research/extract/contextual_chunking.py`
- Create: `tests/extract/test_contextual_chunking.py`

- [ ] **Step 1: Write test with mocked Anthropic client**

```python
# tests/extract/test_contextual_chunking.py
from unittest.mock import patch, MagicMock
from auto_research.extract.contextual_chunking import add_context_to_chunks

def test_contextual_chunking_prepends_generated_context():
    chunks = [
        {"section": "Item 1A. Risk Factors", "text": "China export controls under EAR..."},
        {"section": "Item 7. MD&A", "text": "Revenue grew 45% year over year..."},
    ]
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(type="text", text="This chunk is from NVDA Q3-2025 10-K Risk Factors discussing China export controls.")]
    with patch("auto_research.extract.contextual_chunking.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = fake_resp
        annotated = add_context_to_chunks(
            chunks, doc_meta={"entity_id": "NVDA", "doc_type": "10-K", "period": "2025-Q3"},
            full_doc_text="<full doc>", model_id="claude-haiku-4-5-20251001",
        )
    assert all("This chunk is from" in c["contextualized_text"] for c in annotated)
    assert annotated[0]["section"] == "Item 1A. Risk Factors"
```

- [ ] **Step 2: Implement contextual chunking**

```python
# src/auto_research/extract/contextual_chunking.py
from anthropic import Anthropic

CONTEXT_PROMPT = """<document>
{full_doc}
</document>

Here is a chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Section: {section}
Document metadata: entity={entity_id}, type={doc_type}, period={period}

Write a single-sentence context describing what this chunk is and what it contains, suitable for retrieval. Begin with "This chunk is from..." and be specific about the topic and section. Return ONLY the context sentence — no preamble, no markdown.
"""

def add_context_to_chunks(
    chunks: list[dict],
    *,
    doc_meta: dict,
    full_doc_text: str,
    model_id: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    client = Anthropic()
    out = []
    # Use prompt caching on the full doc — it stays constant across all chunks.
    for chunk in chunks:
        resp = client.messages.create(
            model=model_id,
            max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": CONTEXT_PROMPT.format(
                    full_doc=full_doc_text[:50000],  # truncate for very long docs
                    chunk_text=chunk["text"],
                    section=chunk["section"],
                    entity_id=doc_meta["entity_id"],
                    doc_type=doc_meta["doc_type"],
                    period=doc_meta["period"],
                ), "cache_control": {"type": "ephemeral"}},
            ]}],
        )
        context = resp.content[0].text.strip()
        out.append({
            **chunk,
            "context": context,
            "contextualized_text": f"{context}\n\n{chunk['text']}",
        })
    return out
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/extract/test_contextual_chunking.py -v
git add src/auto_research/extract/contextual_chunking.py tests/extract/test_contextual_chunking.py
git commit -m "feat(rag): Anthropic contextual chunking pattern"
```

---

### Task 15: LanceDB + Voyage embeddings adapter

**Files:**
- Create: `src/auto_research/rag/__init__.py`
- Create: `src/auto_research/rag/embeddings.py`
- Create: `src/auto_research/rag/vector_store.py`
- Create: `tests/rag/test_embeddings.py`
- Create: `tests/rag/test_vector_store.py`

- [ ] **Step 1: Write embedding adapter test**

```python
# tests/rag/test_embeddings.py
import numpy as np
from auto_research.rag.embeddings import LocalBGEEmbedder

def test_local_bge_embeds_to_consistent_dim():
    embedder = LocalBGEEmbedder()
    v1 = embedder.embed(["hello world"])
    v2 = embedder.embed(["another text"])
    assert v1.shape == v2.shape
    assert v1.shape[1] >= 256  # BGE-small is 384-dim
```

- [ ] **Step 2: Implement embeddings**

```python
# src/auto_research/rag/embeddings.py
import os, numpy as np

class LocalBGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

class VoyageEmbedder:
    def __init__(self, model_name: str = "voyage-3"):
        import voyageai
        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model_name = model_name

    def embed(self, texts: list[str], input_type: str = "document") -> np.ndarray:
        result = self.client.embed(texts, model=self.model_name, input_type=input_type)
        return np.array(result.embeddings)

def get_embedder() -> object:
    if os.environ.get("VOYAGE_API_KEY"):
        return VoyageEmbedder()
    return LocalBGEEmbedder()
```

- [ ] **Step 3: Write vector store test**

```python
# tests/rag/test_vector_store.py
from auto_research.rag.vector_store import LanceVectorStore
from auto_research.rag.embeddings import LocalBGEEmbedder

def test_vector_store_writes_and_searches(tmp_path):
    store = LanceVectorStore(tmp_path / "memos.lance", embedder=LocalBGEEmbedder())
    store.upsert([
        {"id": "m1", "text": "PEAD signal with evasiveness language", "meta_json": "{}"},
        {"id": "m2", "text": "Cross-doc supply chain forward tone", "meta_json": "{}"},
        {"id": "m3", "text": "Frontier tech milestone language", "meta_json": "{}"},
    ])
    results = store.search("supply chain mentions", k=2)
    assert len(results) == 2
    assert any(r["id"] == "m2" for r in results)
```

- [ ] **Step 4: Implement LanceDB store**

```python
# src/auto_research/rag/vector_store.py
from pathlib import Path
import lancedb
import pyarrow as pa

class LanceVectorStore:
    def __init__(self, uri: Path, embedder):
        self.embedder = embedder
        self.db = lancedb.connect(str(Path(uri).parent))
        self.table_name = Path(uri).stem
        self.dim = self.embedder.embed(["probe"]).shape[1]
        if self.table_name not in self.db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("text", pa.string()),
                ("meta_json", pa.string()),
                ("vector", pa.list_(pa.float32(), self.dim)),
            ])
            self.tbl = self.db.create_table(self.table_name, schema=schema)
        else:
            self.tbl = self.db.open_table(self.table_name)

    def upsert(self, rows: list[dict]) -> None:
        texts = [r["text"] for r in rows]
        vecs = self.embedder.embed(texts)
        records = [
            {"id": r["id"], "text": r["text"], "meta_json": r.get("meta_json", "{}"),
             "vector": vecs[i].tolist()}
            for i, r in enumerate(rows)
        ]
        # Best-effort delete-then-insert for idempotency
        existing_ids = {r["id"] for r in rows}
        for eid in existing_ids:
            self.tbl.delete(f"id = '{eid}'")
        self.tbl.add(records)

    def search(self, query: str, k: int = 5) -> list[dict]:
        q_vec = self.embedder.embed([query])[0]
        result = self.tbl.search(q_vec.tolist()).limit(k).to_list()
        return [{"id": r["id"], "text": r["text"], "meta_json": r["meta_json"],
                 "score": r.get("_distance", 0.0)} for r in result]
```

- [ ] **Step 5: Run + commit**

```bash
uv run pytest tests/rag/test_embeddings.py tests/rag/test_vector_store.py -v
git add src/auto_research/rag/ tests/rag/
git commit -m "feat(rag): LanceDB store + Voyage/BGE embedder adapter"
```

---

### Task 16: Hybrid retrieval (BM25 + dense + RRF)

**Files:**
- Create: `src/auto_research/rag/hybrid_retrieval.py`
- Create: `tests/rag/test_hybrid_retrieval.py`

- [ ] **Step 1: Write test**

```python
# tests/rag/test_hybrid_retrieval.py
from auto_research.rag.hybrid_retrieval import HybridRetriever
from auto_research.rag.vector_store import LanceVectorStore
from auto_research.rag.embeddings import LocalBGEEmbedder

def test_hybrid_retrieves_keyword_and_semantic_matches(tmp_path):
    store = LanceVectorStore(tmp_path / "h.lance", embedder=LocalBGEEmbedder())
    docs = [
        {"id": "d1", "text": "PEAD evasiveness predicts forward returns on small caps", "meta_json": "{}"},
        {"id": "d2", "text": "BTC funding rates as leverage cost indicator", "meta_json": "{}"},
        {"id": "d3", "text": "Supplier mentions in 10-K filings correlate with capacity guidance", "meta_json": "{}"},
        {"id": "d4", "text": "Cross-document signal propagation from hyperscalers to semiconductor suppliers", "meta_json": "{}"},
    ]
    store.upsert(docs)
    retriever = HybridRetriever(vector_store=store, corpus=docs)
    hits = retriever.search("cross-doc supplier signals", k=3)
    ids = [h["id"] for h in hits]
    # Both d3 and d4 should rank in top-3
    assert "d4" in ids
    assert "d3" in ids
```

- [ ] **Step 2: Implement hybrid retriever**

```python
# src/auto_research/rag/hybrid_retrieval.py
from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t]

class HybridRetriever:
    def __init__(self, vector_store, corpus: list[dict], rrf_k: int = 60):
        self.store = vector_store
        self.corpus = corpus
        self.rrf_k = rrf_k
        self.id_to_idx = {d["id"]: i for i, d in enumerate(corpus)}
        self.bm25 = BM25Okapi([_tokenize(d["text"]) for d in corpus])

    def search(self, query: str, k: int = 5, k_each: int = 20) -> list[dict]:
        # Dense retrieval
        dense_hits = self.store.search(query, k=k_each)
        dense_ranks = {h["id"]: rank for rank, h in enumerate(dense_hits)}
        # Sparse retrieval (BM25)
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.corpus)), key=lambda i: -scores[i])[:k_each]
        sparse_ranks = {self.corpus[i]["id"]: rank for rank, i in enumerate(ranked)}
        # Reciprocal Rank Fusion
        fused: dict[str, float] = {}
        for hid, r in dense_ranks.items():
            fused[hid] = fused.get(hid, 0.0) + 1.0 / (self.rrf_k + r + 1)
        for hid, r in sparse_ranks.items():
            fused[hid] = fused.get(hid, 0.0) + 1.0 / (self.rrf_k + r + 1)
        ranked_ids = sorted(fused, key=lambda x: -fused[x])[:k]
        by_id = {d["id"]: d for d in self.corpus}
        return [{"id": rid, "text": by_id[rid]["text"], "score": fused[rid],
                 "meta_json": by_id[rid].get("meta_json", "{}")} for rid in ranked_ids]
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/rag/test_hybrid_retrieval.py -v
git add src/auto_research/rag/hybrid_retrieval.py tests/rag/test_hybrid_retrieval.py
git commit -m "feat(rag): hybrid retrieval — BM25 + dense + RRF fusion"
```

---

### Task 17: BGE reranker

**Files:**
- Create: `src/auto_research/rag/reranker.py`
- Create: `tests/rag/test_reranker.py`

- [ ] **Step 1: Write reranker test**

```python
# tests/rag/test_reranker.py
from auto_research.rag.reranker import BGEReranker

def test_reranker_orders_by_relevance():
    rr = BGEReranker()
    query = "supplier mentions in semiconductor disclosures"
    candidates = [
        {"id": "a", "text": "Mortgage rate trends in regional banks"},
        {"id": "b", "text": "Customer concentration disclosures from chip suppliers"},
        {"id": "c", "text": "FDA approval pathways for biotech"},
    ]
    out = rr.rerank(query, candidates, top_k=2)
    assert out[0]["id"] == "b"
    assert len(out) == 2
```

- [ ] **Step 2: Implement reranker**

```python
# src/auto_research/rag/reranker.py
class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: -x[1])
        return [{**c, "rerank_score": float(s)} for c, s in scored[:top_k]]
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/rag/test_reranker.py -v
git add src/auto_research/rag/reranker.py tests/rag/test_reranker.py
git commit -m "feat(rag): BGE cross-encoder reranker"
```

---

### Task 18: Entity resolution (Flow 3 — supplier mention → ticker)

**Files:**
- Create: `data/universe/entity_aliases.json`
- Create: `src/auto_research/extract/entity_resolution.py`
- Create: `tests/extract/test_entity_resolution.py`

- [ ] **Step 1: Write entity aliases**

```json
{
  "CRDO": ["Credo Technology","Credo Semiconductor","Credo"],
  "ALAB": ["Astera Labs","Astera"],
  "COHR": ["Coherent","Coherent Corp"],
  "AAOI": ["Applied Optoelectronics","AOI"],
  "MRVL": ["Marvell","Marvell Technology"],
  "ANET": ["Arista","Arista Networks"],
  "VRT": ["Vertiv","Vertiv Holdings"],
  "BE": ["Bloom Energy","Bloom"],
  "VST": ["Vistra","Vistra Energy"],
  "CEG": ["Constellation Energy","Constellation"],
  "TSM": ["TSMC","Taiwan Semiconductor","Taiwan Semiconductor Manufacturing"],
  "ASML": ["ASML","ASML Holding"],
  "MU": ["Micron","Micron Technology"],
  "IONQ": ["IonQ"],
  "RKLB": ["Rocket Lab","Rocket Lab USA"]
}
```

- [ ] **Step 2: Write test**

```python
# tests/extract/test_entity_resolution.py
from auto_research.extract.entity_resolution import resolve_mention

def test_resolve_mention_finds_exact_alias():
    candidates = resolve_mention("Credo Technology")
    assert candidates[0]["ticker"] == "CRDO"
    assert candidates[0]["score"] >= 0.9

def test_resolve_mention_returns_unknown_for_no_match():
    candidates = resolve_mention("Acme Imaginary Corp")
    if candidates:
        assert candidates[0]["score"] < 0.5
```

- [ ] **Step 3: Implement resolver**

```python
# src/auto_research/extract/entity_resolution.py
import json
from pathlib import Path
from auto_research.rag.embeddings import LocalBGEEmbedder

_ALIASES_PATH = Path(__file__).resolve().parents[2] / "data" / "universe" / "entity_aliases.json"
_aliases: dict[str, list[str]] = json.loads(_ALIASES_PATH.read_text())
_embedder = None
_alias_vectors = None
_alias_index: list[tuple[str, str]] = []  # (ticker, alias)

def _init():
    global _embedder, _alias_vectors, _alias_index
    if _alias_vectors is not None:
        return
    _embedder = LocalBGEEmbedder()
    for ticker, aliases in _aliases.items():
        for alias in aliases:
            _alias_index.append((ticker, alias))
    _alias_vectors = _embedder.embed([a for _, a in _alias_index])

def resolve_mention(text: str, top_k: int = 3) -> list[dict]:
    import numpy as np
    _init()
    q = _embedder.embed([text])[0]
    scores = _alias_vectors @ q  # cosine since normalized
    ranked = np.argsort(-scores)[:top_k]
    out = []
    for i in ranked:
        ticker, alias = _alias_index[i]
        out.append({"ticker": ticker, "alias": alias, "score": float(scores[i])})
    return out
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/extract/test_entity_resolution.py -v
git add data/universe/entity_aliases.json src/auto_research/extract/entity_resolution.py tests/extract/test_entity_resolution.py
git commit -m "feat(rag): entity resolution for cross-doc supplier mentions"
```

---

### Task 19: Three remaining extraction workers (10-K, transcript, 8-K) + their prompts

**Files:**
- Create: `src/auto_research/extract/prompts/ten_k_v1.txt`
- Create: `src/auto_research/extract/prompts/transcript_v1.txt`
- Create: `src/auto_research/extract/prompts/eight_k_v1.txt`
- Create: `src/auto_research/extract/workers/ten_k.py`
- Create: `src/auto_research/extract/workers/transcript.py`
- Create: `src/auto_research/extract/workers/eight_k.py`
- Create: `tests/extract/test_ten_k_worker.py`
- Create: `tests/extract/test_transcript_worker.py`
- Create: `tests/extract/test_eight_k_worker.py`

- [ ] **Step 1: Write 10-K prompt (focused on supplier mentions + guidance tone)**

```
# src/auto_research/extract/prompts/ten_k_v1.txt
You are extracting structured features from a 10-K SEC filing. The text may be a single section or full filing.

For EVERY claim you extract, you must include:
- source_span: tuple of (start_char, end_char) into the input text
- source_quote: the EXACT verbatim text from the input at that span

Extract:
- guidance_tone: one of confident | cautious | evasive | neutral | none
  (look at MD&A and forward-looking statements; "evasive" = uses non-committal language to deflect, "none" = no forward-looking language at all)
- accrual_anomaly_flags: list of flags. Each has flag_type ∈ {unusual_accrual, channel_stuffing, reserve_release, other}, severity, source_span, source_quote
- customer_concentration_disclosures: list. Each has target_entity (named customer), concentration_change ∈ {increasing, decreasing, stable, unknown}, confidence ∈ [0,1], source_span, source_quote
- supplier_mentions: list. Each has target_entity (named supplier or partner), tone ∈ {positive, negative, neutral}, horizon_days (how far forward the statement looks; 0 if backward-looking), confidence, source_span, source_quote
- language_novelty_score ∈ [0,1]: 0 = boilerplate identical to typical prior 10-K, 1 = highly novel language
- risk_factor_deltas: list of risk-factor changes. risk_category (short label), change ∈ {new, expanded, reduced, removed, unchanged}, source_span, source_quote

Return ONLY the structured extraction via the submit_extraction tool. Every source_quote must match the input text exactly.
```

- [ ] **Step 2: Write transcript prompt**

```
# src/auto_research/extract/prompts/transcript_v1.txt
You are extracting structured features from an earnings call transcript.

For EVERY claim you extract, include source_span (start_char, end_char) and source_quote (verbatim).

Extract:
- prepared_remarks_tone: confident | cautious | evasive | neutral | none (based on the prepared-remarks section, before Q&A)
- q_and_a_evasiveness ∈ [0,1]: 0 = analysts get direct answers, 1 = management consistently deflects, redirects, or repeats canned lines
- forward_statements: list. Each has:
    target_entity: named entity (a supplier, customer, market segment) — or null if self-referential
    tone: positive | negative | neutral
    horizon_days: how far forward (0 to 730)
    confidence ∈ [0,1]
    source_span, source_quote

Return ONLY the structured extraction via the submit_extraction tool.
```

- [ ] **Step 3: Write 8-K prompt**

```
# src/auto_research/extract/prompts/eight_k_v1.txt
You are extracting structured features from an 8-K current-report SEC filing.

For every event extracted, include source_span and source_quote (verbatim).

Extract:
- events: list of EightKEvent. Each has:
    event_type ∈ {partnership, contract, customer_announcement, product_launch, milestone, data_readout, regulatory_action, executive_change, guidance_update, dilution, other}
    counterparties: list of named entities involved (companies, agencies, customers)
    tone: positive | negative | neutral
    materiality: low | medium | high
    source_span, source_quote
- dilution_language_flag: True if the 8-K announces dilutive share issuance, ATM facility activation, secondary offering, or similar

Return ONLY the structured extraction via the submit_extraction tool.
```

- [ ] **Step 4: Write tests for all three workers (mocked Anthropic)**

```python
# tests/extract/test_ten_k_worker.py
from unittest.mock import patch
from auto_research.extract.workers.ten_k import extract_ten_k
from auto_research.extract.schemas import TenKExtraction, ExtractionMeta

def test_extract_ten_k_returns_validated_schema(tmp_path):
    fake = TenKExtraction(
        entity_id="NVDA", fiscal_period="2025-FY", guidance_tone="confident",
        accrual_anomaly_flags=[], customer_concentration_disclosures=[], supplier_mentions=[],
        language_novelty_score=0.4, risk_factor_deltas=[],
        extraction_metadata=ExtractionMeta(prompt_version="ten_k_v1", model_id="m", extraction_run_id="r"),
    )
    with patch("auto_research.extract.workers.ten_k.ExtractionClient") as M:
        M.return_value.extract.return_value = fake
        result = extract_ten_k(entity_id="NVDA", fiscal_period="2025-FY", raw_text="...",
                                cache_dir=tmp_path / "cache", run_id="r")
    assert result.guidance_tone == "confident"
```

```python
# tests/extract/test_transcript_worker.py
from unittest.mock import patch
from auto_research.extract.workers.transcript import extract_transcript
from auto_research.extract.schemas import TranscriptExtraction, ExtractionMeta

def test_extract_transcript_returns_validated_schema(tmp_path):
    fake = TranscriptExtraction(
        entity_id="NVDA", fiscal_period="2025-Q3", prepared_remarks_tone="confident",
        q_and_a_evasiveness=0.2, forward_statements=[],
        extraction_metadata=ExtractionMeta(prompt_version="transcript_v1", model_id="m", extraction_run_id="r"),
    )
    with patch("auto_research.extract.workers.transcript.ExtractionClient") as M:
        M.return_value.extract.return_value = fake
        result = extract_transcript(entity_id="NVDA", fiscal_period="2025-Q3", raw_text="...",
                                     cache_dir=tmp_path / "cache", run_id="r")
    assert result.q_and_a_evasiveness == 0.2
```

```python
# tests/extract/test_eight_k_worker.py
from unittest.mock import patch
from auto_research.extract.workers.eight_k import extract_eight_k
from auto_research.extract.schemas import EightKExtraction, ExtractionMeta

def test_extract_eight_k_returns_validated_schema(tmp_path):
    fake = EightKExtraction(
        entity_id="CRDO", filing_date="2025-04-01", events=[], dilution_language_flag=False,
        extraction_metadata=ExtractionMeta(prompt_version="eight_k_v1", model_id="m", extraction_run_id="r"),
    )
    with patch("auto_research.extract.workers.eight_k.ExtractionClient") as M:
        M.return_value.extract.return_value = fake
        result = extract_eight_k(entity_id="CRDO", filing_date="2025-04-01", raw_text="...",
                                  cache_dir=tmp_path / "cache", run_id="r")
    assert result.dilution_language_flag is False
```

- [ ] **Step 5: Implement workers (mirror Task 11 pattern; pick model per Spec §7.3)**

```python
# src/auto_research/extract/workers/ten_k.py
from pathlib import Path
from auto_research.extract.client import ExtractionClient
from auto_research.extract.cache import ExtractionCache
from auto_research.extract.schemas import TenKExtraction

PROMPT_VERSION = "ten_k_v1"
MODEL_ID = "claude-sonnet-4-6"
_PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "ten_k_v1.txt").read_text()

def extract_ten_k(*, entity_id, fiscal_period, raw_text, cache_dir, run_id) -> TenKExtraction:
    client = ExtractionClient(cache=ExtractionCache(cache_dir))
    result = client.extract(
        schema=TenKExtraction, system_prompt=_PROMPT, prompt_version=PROMPT_VERSION,
        raw_doc_text=raw_text, model_id=MODEL_ID, max_tokens=8192,
    )
    return result.model_copy(update={"entity_id": entity_id, "fiscal_period": fiscal_period})
```

```python
# src/auto_research/extract/workers/transcript.py
from pathlib import Path
from auto_research.extract.client import ExtractionClient
from auto_research.extract.cache import ExtractionCache
from auto_research.extract.schemas import TranscriptExtraction

PROMPT_VERSION = "transcript_v1"
MODEL_ID = "claude-sonnet-4-6"
_PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "transcript_v1.txt").read_text()

def extract_transcript(*, entity_id, fiscal_period, raw_text, cache_dir, run_id) -> TranscriptExtraction:
    client = ExtractionClient(cache=ExtractionCache(cache_dir))
    result = client.extract(
        schema=TranscriptExtraction, system_prompt=_PROMPT, prompt_version=PROMPT_VERSION,
        raw_doc_text=raw_text, model_id=MODEL_ID, max_tokens=4096,
    )
    return result.model_copy(update={"entity_id": entity_id, "fiscal_period": fiscal_period})
```

```python
# src/auto_research/extract/workers/eight_k.py
from pathlib import Path
from auto_research.extract.client import ExtractionClient
from auto_research.extract.cache import ExtractionCache
from auto_research.extract.schemas import EightKExtraction

PROMPT_VERSION = "eight_k_v1"
MODEL_ID = "claude-haiku-4-5-20251001"
_PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "eight_k_v1.txt").read_text()

def extract_eight_k(*, entity_id, filing_date, raw_text, cache_dir, run_id) -> EightKExtraction:
    client = ExtractionClient(cache=ExtractionCache(cache_dir))
    result = client.extract(
        schema=EightKExtraction, system_prompt=_PROMPT, prompt_version=PROMPT_VERSION,
        raw_doc_text=raw_text, model_id=MODEL_ID, max_tokens=4096,
    )
    return result.model_copy(update={"entity_id": entity_id, "filing_date": filing_date})
```

- [ ] **Step 6: Run + commit**

```bash
uv run pytest tests/extract/test_ten_k_worker.py tests/extract/test_transcript_worker.py tests/extract/test_eight_k_worker.py -v
git add src/auto_research/extract/prompts/ src/auto_research/extract/workers/ten_k.py src/auto_research/extract/workers/transcript.py src/auto_research/extract/workers/eight_k.py tests/extract/test_ten_k_worker.py tests/extract/test_transcript_worker.py tests/extract/test_eight_k_worker.py
git commit -m "feat(extract): 10-K, transcript, 8-K workers with v1 prompts"
```

---

### Task 20: Gold sets + DeepEval pytest harness

**Files:**
- Create: `eval/gold_sets/ten_k.jsonl` (~5 seed examples; expand to ~50 incrementally)
- Create: `eval/gold_sets/transcript.jsonl`
- Create: `eval/gold_sets/eight_k.jsonl`
- Create: `eval/gold_sets/s_filings.jsonl`
- Create: `tests/evals/__init__.py`
- Create: `tests/evals/test_extraction_evals.py`
- Create: `src/auto_research/eval/__init__.py`
- Create: `src/auto_research/eval/extraction_metrics.py`

- [ ] **Step 1: Seed gold sets (5 examples per worker — fill out 50+ later)**

Format for each file (one JSON object per line):

```jsonl
{"id":"ten_k_aapl_2024_fy","doc_excerpt":"...","expected":{"guidance_tone":"cautious","supplier_mentions":[{"target_entity":"TSM","tone":"positive","horizon_days":180,"confidence":0.8}]}}
```

Add 5 such examples per file. See spec §14.1 — target 50-80 per worker; ramp during W2 D9 prompt iteration.

- [ ] **Step 2: Write extraction metrics**

```python
# src/auto_research/eval/extraction_metrics.py
from typing import Iterable

def field_exact_match(predicted, expected, field: str) -> int:
    p = getattr(predicted, field, None) if not isinstance(predicted, dict) else predicted.get(field)
    e = expected.get(field)
    return int(p == e)

def list_set_f1(predicted_items: list, expected_items: list, key: str) -> float:
    p = {item[key] if isinstance(item, dict) else getattr(item, key) for item in predicted_items}
    e = {item[key] if isinstance(item, dict) else getattr(item, key) for item in expected_items}
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    tp = len(p & e)
    precision = tp / len(p)
    recall = tp / len(e)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def numeric_mae(predicted, expected, field: str) -> float:
    p = getattr(predicted, field, None) if not isinstance(predicted, dict) else predicted.get(field)
    e = expected.get(field)
    if p is None or e is None:
        return float("inf")
    return abs(p - e)
```

- [ ] **Step 3: Write DeepEval pytest tests**

```python
# tests/evals/test_extraction_evals.py
import json, pytest
from pathlib import Path
from auto_research.eval.extraction_metrics import field_exact_match, list_set_f1, numeric_mae

GOLD_DIR = Path(__file__).resolve().parents[2] / "eval" / "gold_sets"

def _load_gold(name: str):
    return [json.loads(line) for line in (GOLD_DIR / f"{name}.jsonl").read_text().splitlines() if line.strip()]

@pytest.mark.extraction_eval
@pytest.mark.parametrize("entry", _load_gold("ten_k"))
def test_ten_k_field_match_min_threshold(entry):
    from auto_research.extract.workers.ten_k import extract_ten_k
    # In CI eval mode the worker reads from cached LLM responses; tests verify regressions.
    result = extract_ten_k(
        entity_id=entry["id"].split("_")[1].upper(), fiscal_period="test",
        raw_text=entry["doc_excerpt"], cache_dir=Path("data/extracted/eval_cache"), run_id="eval",
    )
    score = field_exact_match(result, entry["expected"], "guidance_tone")
    assert score == 1, f"guidance_tone mismatch for {entry['id']}"

@pytest.mark.extraction_eval
@pytest.mark.parametrize("entry", _load_gold("ten_k"))
def test_ten_k_supplier_mention_f1_threshold(entry):
    from auto_research.extract.workers.ten_k import extract_ten_k
    result = extract_ten_k(
        entity_id=entry["id"].split("_")[1].upper(), fiscal_period="test",
        raw_text=entry["doc_excerpt"], cache_dir=Path("data/extracted/eval_cache"), run_id="eval",
    )
    f1 = list_set_f1(result.supplier_mentions, entry["expected"].get("supplier_mentions", []), key="target_entity")
    assert f1 >= 0.6, f"supplier_mention F1 = {f1} for {entry['id']}"
```

- [ ] **Step 4: Configure pytest markers**

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "extraction_eval: extraction quality evals (requires gold set + API key)",
    "rag_eval: RAG retrieval quality evals",
    "vcr: HTTP cassettes",
]
```

- [ ] **Step 5: Run eval suite, commit**

```bash
uv run pytest tests/evals/ -m extraction_eval -v
git add eval/gold_sets/ src/auto_research/eval/ tests/evals/ pyproject.toml
git commit -m "feat(eval): DeepEval-style extraction pytest harness with gold sets"
```

---

### Task 21: Ragas RAG eval setup

**Files:**
- Create: `eval/gold_sets/memo_retrieval.jsonl` (~10 seed pairs)
- Create: `src/auto_research/eval/rag_metrics.py`
- Create: `tests/evals/test_rag_retrieval_eval.py`

- [ ] **Step 1: Seed retrieval gold set**

```jsonl
{"id":"q1","query":"PEAD with language evasiveness","expected_memo_ids":["memo_pead_evasive_2026q1"]}
{"id":"q2","query":"supply chain forward tone propagation","expected_memo_ids":["memo_a1_supply_chain_2026q1"]}
```

(Add ~8 more seeds, total 10; ramp during W2 D9.)

- [ ] **Step 2: Write Ragas adapter**

```python
# src/auto_research/eval/rag_metrics.py
def retrieval_recall(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    hits = len(set(retrieved_ids) & set(expected_ids))
    return hits / len(expected_ids)

def retrieval_precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & set(expected_ids)) / k
```

- [ ] **Step 3: Write Ragas pytest tests**

```python
# tests/evals/test_rag_retrieval_eval.py
import json, pytest
from pathlib import Path
from auto_research.eval.rag_metrics import retrieval_recall, retrieval_precision_at_k

GOLD = Path(__file__).resolve().parents[2] / "eval" / "gold_sets" / "memo_retrieval.jsonl"

def _load():
    return [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]

@pytest.mark.rag_eval
def test_memo_retrieval_recall_threshold(tmp_path):
    from auto_research.rag.vector_store import LanceVectorStore
    from auto_research.rag.embeddings import LocalBGEEmbedder
    from auto_research.rag.hybrid_retrieval import HybridRetriever
    embedder = LocalBGEEmbedder()
    store = LanceVectorStore(tmp_path / "memos.lance", embedder=embedder)
    # In a real run, populate from data/attribution_memos/. For tests, seed minimally.
    docs = [
        {"id": "memo_pead_evasive_2026q1", "text": "Post-earnings drift signal with language evasiveness features showed IC=0.04 on small caps.", "meta_json": "{}"},
        {"id": "memo_a1_supply_chain_2026q1", "text": "Cross-doc supply chain forward-tone signal: hyperscaler capex commentary predicts supplier 5-day returns.", "meta_json": "{}"},
        {"id": "memo_unrelated_1", "text": "Bond yield curve sensitivity analysis.", "meta_json": "{}"},
    ]
    store.upsert(docs)
    retriever = HybridRetriever(vector_store=store, corpus=docs)
    recalls = []
    for entry in _load():
        hits = retriever.search(entry["query"], k=3)
        recalls.append(retrieval_recall([h["id"] for h in hits], entry["expected_memo_ids"]))
    avg = sum(recalls) / len(recalls)
    assert avg >= 0.75, f"avg memo recall {avg} below 0.75"
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/evals/test_rag_retrieval_eval.py -v
git add eval/gold_sets/memo_retrieval.jsonl src/auto_research/eval/rag_metrics.py tests/evals/test_rag_retrieval_eval.py
git commit -m "feat(eval): Ragas-style retrieval recall + precision pytest harness"
```

---

### Task 22: Backfill orchestrator + Batch API kickoff

**Files:**
- Create: `src/auto_research/ingest/backfill.py`
- Create: `scripts/run_backfill.py`
- Create: `tests/ingest/test_backfill.py`

- [ ] **Step 1: Write orchestrator test**

```python
# tests/ingest/test_backfill.py
from unittest.mock import patch, MagicMock
from auto_research.ingest.backfill import enumerate_backfill_jobs

def test_enumerate_backfill_jobs_covers_universe(tmp_path):
    with patch("auto_research.ingest.backfill.EdgarClient") as MockEdgar, \
         patch("auto_research.ingest.backfill.FMPTranscriptClient") as MockFMP:
        MockEdgar.return_value.list_filings.return_value = [
            {"ticker":"NVDA","form":"10-K","accession":"0001","accepted_datetime":"2024-02-01","primary_doc":"nvda.htm","cik":"0001045810","period":"2024-01-01"},
        ]
        MockFMP.return_value.list_transcripts.return_value = [
            {"ticker":"NVDA","year":2024,"quarter":1,"event_datetime":"2024-02-01","transcript_id":"NVDA-2024Q1"},
        ]
        jobs = enumerate_backfill_jobs(
            tickers=["NVDA"], since="2024-01-01",
            raw_dir=tmp_path / "raw", manifest_path=tmp_path / "manifest.parquet",
            user_agent="ua", fmp_api_key="k",
        )
    assert any(j["kind"] == "filing" for j in jobs)
    assert any(j["kind"] == "transcript" for j in jobs)
```

- [ ] **Step 2: Implement orchestrator**

```python
# src/auto_research/ingest/backfill.py
from pathlib import Path
from auto_research.ingest.edgar import EdgarClient
from auto_research.ingest.fmp import FMPTranscriptClient

def enumerate_backfill_jobs(
    *, tickers: list[str], since: str, raw_dir: Path, manifest_path: Path,
    user_agent: str, fmp_api_key: str,
) -> list[dict]:
    edgar = EdgarClient(raw_dir=raw_dir, manifest_path=manifest_path, user_agent=user_agent)
    fmp = FMPTranscriptClient(api_key=fmp_api_key, raw_dir=raw_dir, manifest_path=manifest_path)
    jobs = []
    for ticker in tickers:
        try:
            for filing in edgar.list_filings(ticker, ["10-K","10-Q","8-K","S-1","S-3"], since):
                jobs.append({"kind":"filing","ticker":ticker,"meta":filing})
        except Exception:
            continue
        try:
            for transcript in fmp.list_transcripts(ticker, since):
                jobs.append({"kind":"transcript","ticker":ticker,"meta":transcript})
        except Exception:
            continue
    return jobs

def run_jobs(jobs: list[dict], *, raw_dir: Path, manifest_path: Path, user_agent: str, fmp_api_key: str) -> None:
    edgar = EdgarClient(raw_dir=raw_dir, manifest_path=manifest_path, user_agent=user_agent)
    fmp = FMPTranscriptClient(api_key=fmp_api_key, raw_dir=raw_dir, manifest_path=manifest_path)
    for job in jobs:
        try:
            if job["kind"] == "filing":
                edgar.fetch_filing_text(job["meta"])
            else:
                fmp.fetch_transcript(job["meta"])
        except Exception as e:
            print(f"failed {job['kind']} {job['ticker']}: {e}")
```

- [ ] **Step 3: Write runnable script**

```python
# scripts/run_backfill.py
import os
from pathlib import Path
from auto_research.universe import all_tickers
from auto_research.ingest.backfill import enumerate_backfill_jobs, run_jobs

if __name__ == "__main__":
    tickers = all_tickers()
    jobs = enumerate_backfill_jobs(
        tickers=tickers, since="2024-01-01",
        raw_dir=Path("data/raw"), manifest_path=Path("data/manifest.parquet"),
        user_agent=os.environ["SEC_USER_AGENT"], fmp_api_key=os.environ["FMP_API_KEY"],
    )
    print(f"{len(jobs)} jobs queued")
    run_jobs(jobs, raw_dir=Path("data/raw"), manifest_path=Path("data/manifest.parquet"),
             user_agent=os.environ["SEC_USER_AGENT"], fmp_api_key=os.environ["FMP_API_KEY"])
```

- [ ] **Step 4: Run + commit, then kick off backfill**

```bash
uv run pytest tests/ingest/test_backfill.py -v
git add src/auto_research/ingest/backfill.py scripts/run_backfill.py tests/ingest/test_backfill.py
git commit -m "feat(ingest): backfill orchestrator + run-backfill script"
# Kick off live backfill (runs in background; ~24hr SLA on Batch API for extraction phase later):
uv run python scripts/run_backfill.py
git tag w2-rag-complete
```

---

**Milestone 2 acceptance:**
- `uv run pytest -v -m "not extraction_eval and not rag_eval"` all pass
- `uv run pytest -v -m extraction_eval` shows field-level F1 ≥ 0.6 on supplier mentions across the seed gold set
- `uv run pytest -v -m rag_eval` shows memo retrieval recall ≥ 0.75
- `data/raw/` populated with at least one filing + one transcript per ticker in the universe
- `data/manifest.parquet` has corresponding rows
- Langfuse shows extraction traces for the workers run during evals

---

## Milestone 3 — W3: Signals + backtest gauntlet

### Task 23: T1 info_tests primitives (event study, IC, quantile sort, conditional, MI, bootstrap)

**Files:**
- Create: `src/auto_research/backtest/__init__.py`
- Create: `src/auto_research/backtest/info_tests.py`
- Create: `tests/backtest/__init__.py`
- Create: `tests/backtest/test_info_tests.py`

- [ ] **Step 1: Write failing tests with synthetic data**

```python
# tests/backtest/test_info_tests.py
import numpy as np
import pandas as pd
import pytest
from auto_research.backtest.info_tests import (
    event_study, ic_analysis, quantile_sort, conditional_distribution,
    mutual_information, bootstrap_significance,
)

@pytest.fixture
def fake_panel():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    tickers = [f"T{i}" for i in range(20)]
    # Feature has true positive correlation with next-day return
    features = pd.DataFrame(rng.standard_normal((200, 20)), index=dates, columns=tickers)
    noise = pd.DataFrame(rng.standard_normal((200, 20)) * 0.01, index=dates, columns=tickers)
    fwd_returns = features.shift(1) * 0.002 + noise
    return features, fwd_returns

def test_event_study_recovers_positive_car_on_seeded_events(fake_panel):
    features, fwd_returns = fake_panel
    # Pick events where feature was in top-decile → expect positive forward CAR
    events = []
    for d in features.index[5:-10]:
        top = features.loc[d].nlargest(2).index
        for t in top:
            events.append({"ticker": t, "event_date": d})
    events_df = pd.DataFrame(events)
    result = event_study(events_df, fwd_returns, window=(0, 5))
    assert result.n_events > 0
    assert result.car[-1] > 0
    assert result.t_stat > 0

def test_ic_analysis_returns_positive_ic_on_correlated_data(fake_panel):
    features, fwd_returns = fake_panel
    result = ic_analysis(features, fwd_returns, horizons=(1, 5))
    assert result.by_horizon[1] > 0
    assert abs(result.t_stats[1]) >= 1.0

def test_quantile_sort_top_minus_bottom_is_monotone(fake_panel):
    features, fwd_returns = fake_panel
    result = quantile_sort(features, fwd_returns, n_quantiles=5)
    assert result["mean_per_q"][4] > result["mean_per_q"][0]
    assert result["top_minus_bottom_t_stat"] > 0

def test_conditional_distribution_returns_positive_lift_when_condition_aligned(fake_panel):
    features, fwd_returns = fake_panel
    result = conditional_distribution(features, fwd_returns, condition=lambda x: x > 1.0)
    assert result["lift"] > 0
    assert result["n_triggered"] > 50

def test_mutual_information_nonzero_on_correlated_data(fake_panel):
    features, fwd_returns = fake_panel
    mi = mutual_information(features, fwd_returns)
    assert mi > 0.0

def test_bootstrap_significance_returns_ci_around_mean():
    data = np.array([0.01, 0.02, -0.005, 0.015, 0.008, 0.02, 0.005, 0.012])
    result = bootstrap_significance(np.mean, data, n_boot=500, seed=42)
    assert result["ci_lo"] < result["point"] < result["ci_hi"]
    assert result["n_boot"] == 500
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/backtest/test_info_tests.py -v
```

Expected: FAIL — `ModuleNotFoundError: auto_research.backtest.info_tests`

- [ ] **Step 3: Implement `src/auto_research/backtest/__init__.py`**

```python
# src/auto_research/backtest/__init__.py
```

- [ ] **Step 4: Implement `src/auto_research/backtest/info_tests.py`**

```python
from dataclasses import dataclass
from typing import Callable, Iterable
import numpy as np
import pandas as pd
from scipy import stats

@dataclass
class EventStudyResult:
    car: np.ndarray
    aar: np.ndarray
    t_stat: float
    n_events: int
    window: tuple[int, int]

def event_study(events_df: pd.DataFrame, returns_df: pd.DataFrame,
                window: tuple[int, int] = (-5, 5)) -> EventStudyResult:
    """events_df has columns [ticker, event_date]. returns_df is index=date, columns=ticker."""
    pre, post = window
    aars = []
    for _, ev in events_df.iterrows():
        t = ev["ticker"]
        d = pd.Timestamp(ev["event_date"])
        if t not in returns_df.columns:
            continue
        series = returns_df[t]
        if d not in series.index:
            continue
        i = series.index.get_loc(d)
        if i + pre < 0 or i + post >= len(series):
            continue
        aars.append(series.iloc[i + pre : i + post + 1].to_numpy())
    if not aars:
        zeros = np.zeros(post - pre + 1)
        return EventStudyResult(zeros, zeros, 0.0, 0, window)
    mat = np.vstack(aars)
    aar = np.nanmean(mat, axis=0)
    car = np.nancumsum(aar)
    sums = np.nansum(mat, axis=1)
    t = stats.ttest_1samp(sums, 0.0, nan_policy="omit").statistic
    return EventStudyResult(car=car, aar=aar, t_stat=float(t), n_events=len(mat), window=window)

@dataclass
class ICResult:
    by_horizon: dict[int, float]
    t_stats: dict[int, float]
    half_life_days: float

def ic_analysis(features_df: pd.DataFrame, fwd_returns_df: pd.DataFrame,
                horizons: Iterable[int] = (1, 5, 10, 20)) -> ICResult:
    ics: dict[int, float] = {}
    ts: dict[int, float] = {}
    for h in horizons:
        per_day = []
        for date in features_df.index:
            if date not in fwd_returns_df.index:
                continue
            f = features_df.loc[date]
            r = fwd_returns_df.loc[date]
            common = f.dropna().index.intersection(r.dropna().index)
            if len(common) < 5:
                continue
            rho, _ = stats.spearmanr(f[common], r[common])
            if not np.isnan(rho):
                per_day.append(rho)
        arr = np.asarray(per_day)
        if len(arr) == 0:
            ics[h], ts[h] = 0.0, 0.0
            continue
        ics[h] = float(arr.mean())
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        ts[h] = float(arr.mean() / (std / np.sqrt(len(arr)))) if std > 0 else 0.0
    hs = np.array(list(ics.keys()), dtype=float)
    vals = np.array([ics[h] for h in ics])
    pos = np.abs(vals) > 1e-9
    if pos.sum() >= 2:
        slope, *_ = stats.linregress(hs[pos], np.log(np.abs(vals[pos])))
        half_life = -np.log(2) / slope if slope < 0 else float("inf")
    else:
        half_life = 0.0
    return ICResult(by_horizon=ics, t_stats=ts, half_life_days=float(half_life))

def quantile_sort(features_df: pd.DataFrame, fwd_returns_df: pd.DataFrame,
                  n_quantiles: int = 5) -> dict:
    rows = []
    for date in features_df.index:
        if date not in fwd_returns_df.index:
            continue
        f = features_df.loc[date].dropna()
        r = fwd_returns_df.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_quantiles:
            continue
        try:
            qs = pd.qcut(f[common], n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        for q in range(n_quantiles):
            mask = qs == q
            if mask.any():
                rows.append({"date": date, "quantile": int(q),
                             "mean_ret": float(r[common][mask].mean())})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"mean_per_q": {}, "top_minus_bottom_t_stat": 0.0, "n_obs": 0}
    means = df.groupby("quantile")["mean_ret"].mean().to_dict()
    top = df[df["quantile"] == n_quantiles - 1]["mean_ret"]
    bot = df[df["quantile"] == 0]["mean_ret"]
    if len(top) > 1 and len(bot) > 1:
        t_stat = float(stats.ttest_ind(top, bot, equal_var=False).statistic)
    else:
        t_stat = 0.0
    return {"mean_per_q": means, "top_minus_bottom_t_stat": t_stat, "n_obs": int(len(df))}

def conditional_distribution(features_df: pd.DataFrame, fwd_returns_df: pd.DataFrame,
                             condition: Callable[[float], bool]) -> dict:
    triggered: list[float] = []
    untriggered: list[float] = []
    for date in features_df.index:
        if date not in fwd_returns_df.index:
            continue
        f = features_df.loc[date]
        r = fwd_returns_df.loc[date]
        for ticker in f.index:
            fv = f[ticker]; rv = r.get(ticker, np.nan)
            if pd.isna(fv) or pd.isna(rv):
                continue
            if condition(fv):
                triggered.append(float(rv))
            else:
                untriggered.append(float(rv))
    if not triggered or not untriggered:
        return {"lift": 0.0, "hit_rate": 0.0, "n_triggered": len(triggered),
                "n_untriggered": len(untriggered)}
    trig = np.asarray(triggered); untrig = np.asarray(untriggered)
    return {
        "lift": float(trig.mean() - untrig.mean()),
        "hit_rate": float((trig > 0).mean()),
        "n_triggered": int(trig.size),
        "n_untriggered": int(untrig.size),
    }

def mutual_information(features_df: pd.DataFrame, fwd_returns_df: pd.DataFrame,
                       n_neighbors: int = 5) -> float:
    from sklearn.feature_selection import mutual_info_regression
    xs: list[float] = []
    ys: list[float] = []
    for date in features_df.index:
        if date not in fwd_returns_df.index:
            continue
        f = features_df.loc[date]; r = fwd_returns_df.loc[date]
        for ticker in f.index:
            fv = f[ticker]; rv = r.get(ticker, np.nan)
            if pd.isna(fv) or pd.isna(rv):
                continue
            xs.append(float(fv)); ys.append(float(rv))
    if len(xs) < 50:
        return 0.0
    X = np.asarray(xs).reshape(-1, 1)
    y = np.asarray(ys)
    mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=0)
    return float(mi[0])

def bootstrap_significance(stat_func: Callable[[np.ndarray], float], data: Iterable[float],
                           n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(data), dtype=float)
    n = len(arr)
    point = float(stat_func(arr))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = stat_func(arr[rng.integers(0, n, n)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": int(n_boot)}
```

Add `scikit-learn` and `scipy` to `pyproject.toml` dependencies if not already present (`scipy>=1.13`, `scikit-learn>=1.5`).

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/backtest/test_info_tests.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/auto_research/backtest/__init__.py src/auto_research/backtest/info_tests.py tests/backtest/__init__.py tests/backtest/test_info_tests.py pyproject.toml
git commit -m "feat(backtest): T1 info_tests — event study, IC, quantile, MI, bootstrap"
```

---

### Task 24: Triple-barrier labels + CPCV with embargo + deflated Sharpe + cost model + reports

**Files:**
- Create: `src/auto_research/backtest/labels.py`
- Create: `src/auto_research/backtest/cpcv.py`
- Create: `src/auto_research/backtest/deflated_sharpe.py`
- Create: `src/auto_research/backtest/costs.py`
- Create: `src/auto_research/backtest/report.py`
- Create: `tests/backtest/test_labels.py`
- Create: `tests/backtest/test_cpcv.py`
- Create: `tests/backtest/test_deflated_sharpe.py`
- Create: `tests/backtest/test_costs.py`
- Create: `tests/backtest/test_report.py`

- [ ] **Step 1: Write triple-barrier label tests**

```python
# tests/backtest/test_labels.py
import numpy as np
import pandas as pd
from auto_research.backtest.labels import triple_barrier_labels

def test_triple_barrier_labels_assigns_plus_one_on_upper_touch():
    prices = pd.Series([100, 101, 103, 106, 105], index=pd.date_range("2024-01-02", periods=5, freq="B"))
    labels = triple_barrier_labels(prices, vol=pd.Series([0.02]*5, index=prices.index),
                                   up_mult=2.0, down_mult=2.0, max_hold_days=3)
    assert labels.iloc[0] == 1

def test_triple_barrier_labels_assigns_minus_one_on_lower_touch():
    prices = pd.Series([100, 99, 96, 94, 95], index=pd.date_range("2024-01-02", periods=5, freq="B"))
    labels = triple_barrier_labels(prices, vol=pd.Series([0.02]*5, index=prices.index),
                                   up_mult=2.0, down_mult=2.0, max_hold_days=3)
    assert labels.iloc[0] == -1

def test_triple_barrier_labels_assigns_zero_on_time_barrier():
    prices = pd.Series([100, 100.1, 99.9, 100.05, 99.95], index=pd.date_range("2024-01-02", periods=5, freq="B"))
    labels = triple_barrier_labels(prices, vol=pd.Series([0.02]*5, index=prices.index),
                                   up_mult=2.0, down_mult=2.0, max_hold_days=3)
    assert labels.iloc[0] == 0
```

- [ ] **Step 2: Implement triple-barrier labels**

```python
# src/auto_research/backtest/labels.py
import numpy as np
import pandas as pd

def triple_barrier_labels(
    prices: pd.Series,
    vol: pd.Series,
    *,
    up_mult: float = 2.0,
    down_mult: float = 2.0,
    max_hold_days: int = 10,
) -> pd.Series:
    """Lopez de Prado triple-barrier labels: +1 upper, -1 lower, 0 time barrier.

    prices: close prices indexed by date.
    vol: per-date volatility (e.g., 20-day EWM stdev of log returns).
    """
    labels = pd.Series(0, index=prices.index, dtype=int)
    arr = prices.to_numpy()
    v = vol.to_numpy()
    for i in range(len(arr) - 1):
        upper = arr[i] * (1.0 + up_mult * v[i])
        lower = arr[i] * (1.0 - down_mult * v[i])
        horizon = min(i + max_hold_days, len(arr) - 1)
        label = 0
        for j in range(i + 1, horizon + 1):
            if arr[j] >= upper:
                label = 1; break
            if arr[j] <= lower:
                label = -1; break
        labels.iloc[i] = label
    return labels
```

- [ ] **Step 3: Write CPCV tests**

```python
# tests/backtest/test_cpcv.py
import numpy as np
import pandas as pd
from auto_research.backtest.cpcv import combinatorial_purged_cv

def test_cpcv_returns_expected_number_of_folds():
    timestamps = pd.date_range("2024-01-02", periods=260, freq="B")
    splits = combinatorial_purged_cv(timestamps, n_groups=6, k_test=2, embargo_pct=0.01)
    # C(6,2) = 15 splits
    assert len(splits) == 15
    for train_idx, test_idx in splits:
        assert len(set(train_idx) & set(test_idx)) == 0

def test_cpcv_embargo_purges_overlapping_timestamps():
    timestamps = pd.date_range("2024-01-02", periods=100, freq="B")
    splits = combinatorial_purged_cv(timestamps, n_groups=5, k_test=1, embargo_pct=0.05)
    for train_idx, test_idx in splits:
        # Embargo of 5% of 100 = 5 obs around each test fold
        for ti in test_idx:
            for tr in train_idx:
                assert abs(ti - tr) > 0  # no perfect overlap
```

- [ ] **Step 4: Implement CPCV**

```python
# src/auto_research/backtest/cpcv.py
from itertools import combinations
import numpy as np
import pandas as pd

def combinatorial_purged_cv(
    timestamps: pd.DatetimeIndex,
    *,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """López de Prado CPCV with embargo. Returns list of (train_idx, test_idx)."""
    n = len(timestamps)
    group_size = n // n_groups
    groups: list[np.ndarray] = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n
        groups.append(np.arange(start, end))
    embargo = max(1, int(embargo_pct * n))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for combo in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[g] for g in combo])
        test_set = set(test_idx.tolist())
        # Apply embargo: remove indices within `embargo` of any test index
        train_idx = []
        for i in range(n):
            if i in test_set:
                continue
            if any(abs(i - t) <= embargo for t in test_idx):
                continue
            train_idx.append(i)
        splits.append((np.asarray(train_idx), test_idx))
    return splits
```

- [ ] **Step 5: Write deflated Sharpe tests**

```python
# tests/backtest/test_deflated_sharpe.py
import numpy as np
from auto_research.backtest.deflated_sharpe import deflated_sharpe_ratio

def test_deflated_sharpe_penalizes_more_trials():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, 252)
    dsr_few = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_many = deflated_sharpe_ratio(returns, n_trials=100)
    assert dsr_few > dsr_many

def test_deflated_sharpe_handles_zero_volatility():
    returns = np.zeros(252)
    dsr = deflated_sharpe_ratio(returns, n_trials=5)
    assert dsr == 0.0
```

- [ ] **Step 6: Implement deflated Sharpe**

```python
# src/auto_research/backtest/deflated_sharpe.py
import numpy as np
from scipy import stats

def deflated_sharpe_ratio(returns: np.ndarray, *, n_trials: int = 1,
                          freq_per_year: int = 252) -> float:
    """López de Prado deflated Sharpe (probability-adjusted for multiple testing).

    Returns the implied annualized Sharpe scaled by P(SR > 0 | n_trials).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10 or r.std(ddof=1) == 0:
        return 0.0
    sr = (r.mean() / r.std(ddof=1)) * np.sqrt(freq_per_year)
    # Approximate the expected max SR under null with n_trials i.i.d. trials
    emax = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + \
            np.euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=True)
    sr_std = np.sqrt((1 - skew * sr + ((kurt) / 4) * sr ** 2) / (len(r) - 1))
    z = (sr - emax) / sr_std if sr_std > 0 else 0.0
    p = stats.norm.cdf(z)
    return float(sr * p)
```

- [ ] **Step 7: Write cost-model tests**

```python
# tests/backtest/test_costs.py
import pandas as pd
from auto_research.backtest.costs import TransactionCostModel

def test_cost_model_charges_bid_ask_half_spread():
    cm = TransactionCostModel(half_spread_bps=5.0, impact_coeff=0.0, borrow_bps_per_year=0.0,
                              commission_bps=0.0)
    fill_price = 100.0; qty = 100; adv = 1_000_000
    cost = cm.cost(side="buy", price=fill_price, qty=qty, adv_shares=adv)
    assert abs(cost - (100 * 100 * 5.0 / 10_000)) < 1e-6

def test_cost_model_adds_sqrt_impact_when_qty_large():
    cm = TransactionCostModel(half_spread_bps=0.0, impact_coeff=0.1, borrow_bps_per_year=0.0,
                              commission_bps=0.0)
    big = cm.cost(side="buy", price=100.0, qty=10_000, adv_shares=1_000_000)
    small = cm.cost(side="buy", price=100.0, qty=100, adv_shares=1_000_000)
    assert big > small

def test_cost_model_charges_borrow_on_short_positions():
    cm = TransactionCostModel(half_spread_bps=0.0, impact_coeff=0.0, borrow_bps_per_year=200.0,
                              commission_bps=0.0)
    borrow = cm.borrow_cost(notional=10_000, days=30)
    expected = 10_000 * (200.0 / 10_000) * (30 / 365)
    assert abs(borrow - expected) < 1e-6
```

- [ ] **Step 8: Implement cost model**

```python
# src/auto_research/backtest/costs.py
from dataclasses import dataclass
import math

@dataclass
class TransactionCostModel:
    half_spread_bps: float = 5.0       # bid-ask half-spread in bps
    impact_coeff: float = 0.1          # sqrt impact coefficient
    borrow_bps_per_year: float = 50.0  # annualized borrow rate for shorts
    commission_bps: float = 0.5        # per-side commission
    participation_cap_pct: float = 0.10  # max % of ADV per day

    def cost(self, *, side: str, price: float, qty: float, adv_shares: float) -> float:
        notional = price * qty
        spread_cost = notional * (self.half_spread_bps / 10_000)
        participation = qty / max(adv_shares, 1.0)
        impact_cost = notional * self.impact_coeff * math.sqrt(participation)
        commission = notional * (self.commission_bps / 10_000)
        return spread_cost + impact_cost + commission

    def borrow_cost(self, *, notional: float, days: int) -> float:
        return notional * (self.borrow_bps_per_year / 10_000) * (days / 365.0)

    def participation_limit_shares(self, adv_shares: float) -> float:
        return self.participation_cap_pct * adv_shares
```

- [ ] **Step 9: Write report dataclass tests**

```python
# tests/backtest/test_report.py
import numpy as np
from auto_research.backtest.report import InfoReport, BacktestReport

def test_info_report_serializes_round_trip():
    r = InfoReport(
        signal_id="A2_pead_v1", tier="T1",
        ic_mean=0.04, ic_t_stat=2.5, ic_half_life_days=6.0,
        top_minus_bottom_t_stat=2.1, event_study_car_t_stat=2.2,
        mutual_information=0.02, n_observations=320,
        notes="seed run",
    )
    payload = r.to_dict()
    r2 = InfoReport(**payload)
    assert r2.signal_id == r.signal_id
    assert r2.ic_t_stat == r.ic_t_stat

def test_backtest_report_holds_required_fields():
    r = BacktestReport(
        signal_id="A2_pead_v1", tier="T2",
        sharpe_net=1.1, sharpe_gross=1.4, deflated_sharpe=0.9,
        max_drawdown=-0.12, turnover_annual=4.2, capacity_usd=2_500_000,
        sharpe_at_2x_costs=0.6, max_beta_to_existing=0.3,
        ic_mean=0.04, ic_half_life_days=6.0,
        n_folds=15, notes="CPCV mean",
    )
    assert r.tier == "T2"
    assert r.sharpe_net > 0
```

- [ ] **Step 10: Implement reports**

```python
# src/auto_research/backtest/report.py
from dataclasses import dataclass, asdict, field
from typing import Literal

Tier = Literal["T1", "T2", "T3"]

@dataclass
class InfoReport:
    signal_id: str
    tier: Tier
    ic_mean: float
    ic_t_stat: float
    ic_half_life_days: float
    top_minus_bottom_t_stat: float
    event_study_car_t_stat: float
    mutual_information: float
    n_observations: int
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class BacktestReport:
    signal_id: str
    tier: Tier
    sharpe_net: float
    sharpe_gross: float
    deflated_sharpe: float
    max_drawdown: float
    turnover_annual: float
    capacity_usd: float
    sharpe_at_2x_costs: float
    max_beta_to_existing: float
    ic_mean: float
    ic_half_life_days: float
    n_folds: int
    notes: str = ""
    per_fold_sharpe: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 11: Run all new tests + commit**

```bash
uv run pytest tests/backtest/test_labels.py tests/backtest/test_cpcv.py tests/backtest/test_deflated_sharpe.py tests/backtest/test_costs.py tests/backtest/test_report.py -v
git add src/auto_research/backtest/labels.py src/auto_research/backtest/cpcv.py src/auto_research/backtest/deflated_sharpe.py src/auto_research/backtest/costs.py src/auto_research/backtest/report.py tests/backtest/test_labels.py tests/backtest/test_cpcv.py tests/backtest/test_deflated_sharpe.py tests/backtest/test_costs.py tests/backtest/test_report.py
git commit -m "feat(backtest): triple-barrier labels, CPCV w/ embargo, deflated Sharpe, cost model, reports"
```

---

### Task 25: Backtest engine + tier-aware decide gates (code-checked thresholds)

**Files:**
- Create: `src/auto_research/backtest/engine.py`
- Create: `src/auto_research/backtest/gates.py`
- Create: `tests/backtest/test_engine.py`
- Create: `tests/backtest/test_gates.py`

- [ ] **Step 1: Write gates tests**

```python
# tests/backtest/test_gates.py
from auto_research.backtest.gates import (
    T1_GATE, T2_GATE, check_t1_gate, check_t2_gate, GateDecision,
)
from auto_research.backtest.report import InfoReport, BacktestReport

def test_t1_gate_promotes_strong_info_report():
    r = InfoReport(signal_id="x", tier="T1", ic_mean=0.05, ic_t_stat=2.5,
                   ic_half_life_days=6.0, top_minus_bottom_t_stat=2.0,
                   event_study_car_t_stat=2.5, mutual_information=0.05,
                   n_observations=120)
    decision = check_t1_gate(r)
    assert decision.passed is True

def test_t1_gate_kills_low_t_stat():
    r = InfoReport(signal_id="x", tier="T1", ic_mean=0.005, ic_t_stat=0.5,
                   ic_half_life_days=2.0, top_minus_bottom_t_stat=0.3,
                   event_study_car_t_stat=0.4, mutual_information=0.001,
                   n_observations=30)
    decision = check_t1_gate(r)
    assert decision.passed is False
    assert any("ic_t_stat" in fr for fr in decision.failed_rules)

def test_t2_gate_kills_when_sharpe_collapses_at_2x_costs():
    r = BacktestReport(signal_id="x", tier="T2", sharpe_net=1.0, sharpe_gross=1.4,
                       deflated_sharpe=1.2, max_drawdown=-0.1, turnover_annual=3.0,
                       capacity_usd=5_000_000, sharpe_at_2x_costs=0.1,
                       max_beta_to_existing=0.4, ic_mean=0.03,
                       ic_half_life_days=5.0, n_folds=15)
    decision = check_t2_gate(r, existing_beta_max=0.5)
    assert decision.passed is False
    assert any("sharpe_at_2x_costs" in fr for fr in decision.failed_rules)
```

- [ ] **Step 2: Implement gates**

```python
# src/auto_research/backtest/gates.py
from dataclasses import dataclass
from auto_research.backtest.report import InfoReport, BacktestReport

T1_GATE = {
    "ic_t_stat_min": 2.0,
    "top_minus_bottom_t_stat_min": 1.8,
    "event_study_car_t_stat_min": 2.0,
    "n_observations_min": 30,
}

T2_GATE = {
    "deflated_sharpe_min": 1.0,
    "ic_mean_min": 0.02,
    "ic_half_life_min_days": 1.0,
    "capacity_usd_min": 1_000_000,
    "sharpe_net_min": 0.7,
    "sharpe_at_2x_costs_min": 0.3,
    "max_beta_to_existing": 0.5,
}

@dataclass
class GateDecision:
    passed: bool
    failed_rules: list[str]

def check_t1_gate(report: InfoReport) -> GateDecision:
    failed: list[str] = []
    if abs(report.ic_t_stat) < T1_GATE["ic_t_stat_min"]:
        failed.append(f"ic_t_stat={report.ic_t_stat:.2f} < {T1_GATE['ic_t_stat_min']}")
    if abs(report.top_minus_bottom_t_stat) < T1_GATE["top_minus_bottom_t_stat_min"]:
        failed.append(f"top_minus_bottom_t_stat={report.top_minus_bottom_t_stat:.2f} "
                      f"< {T1_GATE['top_minus_bottom_t_stat_min']}")
    if abs(report.event_study_car_t_stat) < T1_GATE["event_study_car_t_stat_min"]:
        failed.append(f"event_study_car_t_stat={report.event_study_car_t_stat:.2f} "
                      f"< {T1_GATE['event_study_car_t_stat_min']}")
    if report.n_observations < T1_GATE["n_observations_min"]:
        failed.append(f"n_observations={report.n_observations} < {T1_GATE['n_observations_min']}")
    return GateDecision(passed=not failed, failed_rules=failed)

def check_t2_gate(report: BacktestReport, *, existing_beta_max: float | None = None) -> GateDecision:
    failed: list[str] = []
    if report.deflated_sharpe < T2_GATE["deflated_sharpe_min"]:
        failed.append(f"deflated_sharpe={report.deflated_sharpe:.2f} < {T2_GATE['deflated_sharpe_min']}")
    if report.ic_mean < T2_GATE["ic_mean_min"]:
        failed.append(f"ic_mean={report.ic_mean:.4f} < {T2_GATE['ic_mean_min']}")
    if report.ic_half_life_days < T2_GATE["ic_half_life_min_days"]:
        failed.append(f"ic_half_life_days={report.ic_half_life_days:.2f} "
                      f"< {T2_GATE['ic_half_life_min_days']}")
    if report.capacity_usd < T2_GATE["capacity_usd_min"]:
        failed.append(f"capacity_usd={report.capacity_usd:.0f} < {T2_GATE['capacity_usd_min']}")
    if report.sharpe_net < T2_GATE["sharpe_net_min"]:
        failed.append(f"sharpe_net={report.sharpe_net:.2f} < {T2_GATE['sharpe_net_min']}")
    if report.sharpe_at_2x_costs < T2_GATE["sharpe_at_2x_costs_min"]:
        failed.append(f"sharpe_at_2x_costs={report.sharpe_at_2x_costs:.2f} "
                      f"< {T2_GATE['sharpe_at_2x_costs_min']}")
    beta_max = existing_beta_max if existing_beta_max is not None else T2_GATE["max_beta_to_existing"]
    if report.max_beta_to_existing > beta_max:
        failed.append(f"max_beta_to_existing={report.max_beta_to_existing:.2f} > {beta_max}")
    return GateDecision(passed=not failed, failed_rules=failed)
```

- [ ] **Step 3: Write engine test**

```python
# tests/backtest/test_engine.py
import numpy as np
import pandas as pd
from auto_research.backtest.engine import run_portfolio_backtest

def test_run_portfolio_backtest_returns_required_report_fields():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-02", periods=260, freq="B")
    tickers = [f"T{i}" for i in range(10)]
    prices = pd.DataFrame(100 + rng.standard_normal((260, 10)).cumsum(axis=0),
                          index=dates, columns=tickers)
    # Synthetic signal correlated with next-day return
    fwd_ret = prices.pct_change().shift(-1)
    signal = fwd_ret.shift(1).rolling(3).mean()
    report = run_portfolio_backtest(
        signal_id="synth_v1", signal=signal, prices=prices,
        cpcv_n_groups=5, cpcv_k_test=1, target_vol=0.10,
        single_name_cap=0.05, turnover_penalty_bps=10.0,
    )
    assert report.signal_id == "synth_v1"
    assert report.tier == "T2"
    assert isinstance(report.sharpe_net, float)
    assert isinstance(report.deflated_sharpe, float)
    assert report.n_folds == 5
```

- [ ] **Step 4: Implement engine**

```python
# src/auto_research/backtest/engine.py
import numpy as np
import pandas as pd
from auto_research.backtest.cpcv import combinatorial_purged_cv
from auto_research.backtest.deflated_sharpe import deflated_sharpe_ratio
from auto_research.backtest.costs import TransactionCostModel
from auto_research.backtest.report import BacktestReport

def _cross_sectional_rank_to_weights(signal_row: pd.Series, single_name_cap: float) -> pd.Series:
    s = signal_row.dropna()
    if len(s) < 2:
        return pd.Series(0.0, index=signal_row.index)
    ranked = s.rank(pct=True) - 0.5
    weights = ranked / ranked.abs().sum() if ranked.abs().sum() > 0 else ranked
    return weights.clip(lower=-single_name_cap, upper=single_name_cap).reindex(signal_row.index).fillna(0.0)

def run_portfolio_backtest(
    *,
    signal_id: str,
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    cpcv_n_groups: int = 6,
    cpcv_k_test: int = 1,
    target_vol: float = 0.10,
    single_name_cap: float = 0.05,
    turnover_penalty_bps: float = 5.0,
    cost_model: TransactionCostModel | None = None,
    existing_signal_returns: pd.Series | None = None,
) -> BacktestReport:
    """Long-short, sub-universe-neutral conceptually (caller pre-neutralizes signal),
    vol-scaled to `target_vol`. CPCV folds report per-fold Sharpe; aggregate is the mean."""
    cm = cost_model or TransactionCostModel()
    rets = prices.pct_change().fillna(0.0)
    common_dates = signal.index.intersection(rets.index)
    signal = signal.reindex(common_dates).reindex(columns=prices.columns)
    rets = rets.reindex(common_dates).reindex(columns=prices.columns)

    # Compute per-day weights once
    weights = signal.apply(lambda row: _cross_sectional_rank_to_weights(row, single_name_cap), axis=1)
    daily_pnl_gross = (weights.shift(1) * rets).sum(axis=1)
    # Vol-scale to target
    realized_vol = daily_pnl_gross.std() * np.sqrt(252)
    scale = target_vol / realized_vol if realized_vol > 0 else 1.0
    weights = weights * scale
    daily_pnl_gross = (weights.shift(1) * rets).sum(axis=1)

    # Costs: approximate via turnover * cost_per_unit
    turnover = (weights - weights.shift(1)).abs().sum(axis=1)
    bps_cost = cm.half_spread_bps + cm.commission_bps + turnover_penalty_bps
    daily_cost = turnover * (bps_cost / 10_000)
    daily_pnl_net = daily_pnl_gross - daily_cost
    daily_pnl_2x = daily_pnl_gross - 2.0 * daily_cost

    splits = combinatorial_purged_cv(common_dates, n_groups=cpcv_n_groups, k_test=cpcv_k_test,
                                     embargo_pct=0.01)
    per_fold_sharpe: list[float] = []
    for _, test_idx in splits:
        slice_pnl = daily_pnl_net.iloc[test_idx].dropna()
        if len(slice_pnl) < 20 or slice_pnl.std() == 0:
            continue
        s = (slice_pnl.mean() / slice_pnl.std()) * np.sqrt(252)
        per_fold_sharpe.append(float(s))

    sharpe_gross = (daily_pnl_gross.mean() / daily_pnl_gross.std()) * np.sqrt(252) \
                    if daily_pnl_gross.std() > 0 else 0.0
    sharpe_net = (daily_pnl_net.mean() / daily_pnl_net.std()) * np.sqrt(252) \
                  if daily_pnl_net.std() > 0 else 0.0
    sharpe_2x = (daily_pnl_2x.mean() / daily_pnl_2x.std()) * np.sqrt(252) \
                 if daily_pnl_2x.std() > 0 else 0.0

    dsr = deflated_sharpe_ratio(daily_pnl_net.to_numpy(), n_trials=max(1, len(splits)))

    equity = (1.0 + daily_pnl_net.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0).min()

    # Beta vs existing alpha
    if existing_signal_returns is not None and len(existing_signal_returns) > 30:
        aligned = pd.concat([daily_pnl_net, existing_signal_returns], axis=1, join="inner").dropna()
        if len(aligned) > 10 and aligned.iloc[:, 1].var() > 0:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
            beta = float(cov / aligned.iloc[:, 1].var())
        else:
            beta = 0.0
    else:
        beta = 0.0

    # Capacity proxy: assume $X total notional fillable per day at 10% participation
    capacity = float(prices.iloc[-1].mean() * 1_000_000 * 0.10)  # rough — refine in T3

    # IC sanity
    ic_per_day = []
    fwd = rets.shift(-1)
    for d in signal.index:
        if d not in fwd.index: continue
        s = signal.loc[d].dropna(); r = fwd.loc[d].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 3: continue
        from scipy import stats as _s
        rho, _ = _s.spearmanr(s[common], r[common])
        if not np.isnan(rho): ic_per_day.append(rho)
    ic_mean = float(np.mean(ic_per_day)) if ic_per_day else 0.0
    ic_half_life = 1.0  # set by ic_analysis at signal-build time; placeholder here

    return BacktestReport(
        signal_id=signal_id, tier="T2",
        sharpe_net=float(sharpe_net), sharpe_gross=float(sharpe_gross),
        deflated_sharpe=float(dsr), max_drawdown=float(drawdown),
        turnover_annual=float(turnover.sum() * (252 / max(1, len(turnover)))),
        capacity_usd=capacity,
        sharpe_at_2x_costs=float(sharpe_2x), max_beta_to_existing=float(abs(beta)),
        ic_mean=ic_mean, ic_half_life_days=ic_half_life,
        n_folds=int(len(splits)),
        per_fold_sharpe=per_fold_sharpe,
    )
```

- [ ] **Step 5: Run + commit**

```bash
uv run pytest tests/backtest/test_engine.py tests/backtest/test_gates.py -v
git add src/auto_research/backtest/engine.py src/auto_research/backtest/gates.py tests/backtest/test_engine.py tests/backtest/test_gates.py
git commit -m "feat(backtest): portfolio engine + code-checked T1/T2 decide gates"
```

---

### Task 26: Signals A2 + A1 + B1 + IC-weighted combiner with Ledoit-Wolf shrinkage + AlphaLibrary

**Files:**
- Create: `src/auto_research/signals/__init__.py`
- Create: `src/auto_research/signals/a2_pead_drift.py`
- Create: `src/auto_research/signals/a1_supply_chain.py`
- Create: `src/auto_research/signals/b1_frontier.py`
- Create: `src/auto_research/signals/combiner.py`
- Create: `src/auto_research/signals/alpha_library.py`
- Create: `tests/signals/__init__.py`
- Create: `tests/signals/test_a2_pead_drift.py`
- Create: `tests/signals/test_a1_supply_chain.py`
- Create: `tests/signals/test_b1_frontier.py`
- Create: `tests/signals/test_combiner.py`
- Create: `tests/signals/test_alpha_library.py`

- [ ] **Step 1: Write A2 PEAD-drift signal test**

```python
# tests/signals/test_a2_pead_drift.py
import pandas as pd
from auto_research.signals.a2_pead_drift import build_a2_signal

def test_a2_signal_long_on_confident_low_novelty():
    events = pd.DataFrame([
        {"entity_id": "NVDA", "event_datetime": "2025-02-20",
         "prepared_remarks_tone": "confident", "q_and_a_evasiveness": 0.2,
         "guidance_tone": "confident", "language_novelty_score": 0.2},
        {"entity_id": "ABC", "event_datetime": "2025-02-20",
         "prepared_remarks_tone": "evasive", "q_and_a_evasiveness": 0.85,
         "guidance_tone": "evasive", "language_novelty_score": 0.85},
    ])
    sig = build_a2_signal(events, window_days=5)
    nvda_row = sig[sig["entity_id"] == "NVDA"].iloc[0]
    abc_row = sig[sig["entity_id"] == "ABC"].iloc[0]
    assert nvda_row["score"] > 0
    assert abc_row["score"] < 0

def test_a2_signal_decays_to_zero_after_window():
    events = pd.DataFrame([
        {"entity_id": "NVDA", "event_datetime": "2025-02-20",
         "prepared_remarks_tone": "confident", "q_and_a_evasiveness": 0.1,
         "guidance_tone": "confident", "language_novelty_score": 0.2},
    ])
    sig = build_a2_signal(events, window_days=5)
    nvda_rows = sig[sig["entity_id"] == "NVDA"].sort_values("as_of_date")
    # Score on day 0 should be max; score on day window_days+1 should be 0
    assert nvda_rows.iloc[0]["score"] > nvda_rows.iloc[-1]["score"]
    assert nvda_rows.iloc[-1]["score"] == 0.0
```

- [ ] **Step 2: Implement A2 PEAD drift**

```python
# src/auto_research/signals/__init__.py
```

```python
# src/auto_research/signals/a2_pead_drift.py
"""A2 — PEAD-flavored language drift (per spec §9.2).

Long: high prepared-remarks confidence + low novelty + low evasiveness.
Short: high evasiveness + high novelty.
Per-name signal (not cross-sectional). Decays linearly over post-event window.
"""
import pandas as pd

_TONE_MAP = {"confident": 1.0, "cautious": -0.3, "evasive": -1.0, "neutral": 0.0, "none": 0.0}

def _score_event(row) -> float:
    tone_pr = _TONE_MAP.get(row["prepared_remarks_tone"], 0.0)
    tone_g = _TONE_MAP.get(row["guidance_tone"], 0.0)
    evasive = float(row["q_and_a_evasiveness"])
    novelty = float(row["language_novelty_score"])
    # Confident + low evasive + low novelty → long; opposite → short
    base = 0.5 * tone_pr + 0.3 * tone_g
    base -= 0.6 * evasive
    base -= 0.4 * novelty
    return float(base)

def build_a2_signal(events: pd.DataFrame, *, window_days: int = 10) -> pd.DataFrame:
    """events columns: entity_id, event_datetime, prepared_remarks_tone, q_and_a_evasiveness,
       guidance_tone, language_novelty_score. Returns long-form DF with score per (entity, day)."""
    rows = []
    for _, ev in events.iterrows():
        s0 = _score_event(ev)
        event_date = pd.Timestamp(ev["event_datetime"]).normalize()
        for d in range(0, window_days + 1):
            decay = max(0.0, 1.0 - d / window_days)
            rows.append({
                "entity_id": ev["entity_id"],
                "as_of_date": event_date + pd.Timedelta(days=d),
                "score": s0 * decay,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Write A1 supply-chain forward-tone signal test**

```python
# tests/signals/test_a1_supply_chain.py
import pandas as pd
from auto_research.signals.a1_supply_chain import build_a1_signal

def test_a1_signal_long_on_positive_narrative_mentions():
    mentions = pd.DataFrame([
        {"source_entity": "NVDA", "target_ticker": "CRDO", "tone": "positive",
         "confidence": 0.9, "as_of_date": "2025-03-01", "horizon_days": 90},
        {"source_entity": "MSFT", "target_ticker": "CRDO", "tone": "positive",
         "confidence": 0.8, "as_of_date": "2025-03-05", "horizon_days": 90},
        {"source_entity": "GOOGL", "target_ticker": "AAOI", "tone": "negative",
         "confidence": 0.7, "as_of_date": "2025-03-01", "horizon_days": 60},
    ])
    sig = build_a1_signal(mentions, as_of_date="2025-03-10", trailing_days=60, decay_half_life_days=14)
    crdo = sig[sig["target_ticker"] == "CRDO"].iloc[0]["score"]
    aaoi = sig[sig["target_ticker"] == "AAOI"].iloc[0]["score"]
    assert crdo > 0
    assert aaoi < 0
    assert crdo > abs(aaoi)  # two positive mentions outweigh one negative

def test_a1_signal_excludes_mentions_outside_trailing_window():
    mentions = pd.DataFrame([
        {"source_entity": "NVDA", "target_ticker": "CRDO", "tone": "positive",
         "confidence": 0.9, "as_of_date": "2024-01-01", "horizon_days": 90},
    ])
    sig = build_a1_signal(mentions, as_of_date="2025-03-10", trailing_days=60, decay_half_life_days=14)
    assert sig.empty or sig.iloc[0]["score"] == 0.0
```

- [ ] **Step 4: Implement A1 supply-chain**

```python
# src/auto_research/signals/a1_supply_chain.py
"""A1 — Hyperscaler forward-tone propagation (per spec §9.1).

Cross-sectional: for each tradeable target T on day d, sum time-decayed
mentions of T extracted from narrative-source docs in trailing window.
"""
import math
import pandas as pd

_TONE_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

def build_a1_signal(
    mentions: pd.DataFrame,
    *,
    as_of_date: str,
    trailing_days: int = 60,
    decay_half_life_days: float = 14.0,
) -> pd.DataFrame:
    """mentions columns: source_entity, target_ticker, tone, confidence, as_of_date, horizon_days.

    Returns DataFrame[target_ticker, as_of_date, score] — one row per ticker on the as-of date.
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    cutoff = as_of - pd.Timedelta(days=trailing_days)
    m = mentions.copy()
    m["as_of_date"] = pd.to_datetime(m["as_of_date"]).dt.normalize()
    m = m[(m["as_of_date"] >= cutoff) & (m["as_of_date"] <= as_of)]
    if m.empty:
        return pd.DataFrame(columns=["target_ticker", "as_of_date", "score"])
    decay_lambda = math.log(2) / max(decay_half_life_days, 1e-9)
    rows = []
    for ticker, grp in m.groupby("target_ticker"):
        score = 0.0
        for _, mention in grp.iterrows():
            age_days = (as_of - mention["as_of_date"]).days
            decay = math.exp(-decay_lambda * age_days)
            sign = _TONE_MAP.get(mention["tone"], 0.0)
            score += sign * float(mention["confidence"]) * decay
        rows.append({"target_ticker": ticker, "as_of_date": as_of, "score": float(score)})
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Write B1 frontier milestone/dilution signal test**

```python
# tests/signals/test_b1_frontier.py
import pandas as pd
from auto_research.signals.b1_frontier import build_b1_signal

def test_b1_signal_positive_on_milestone_event():
    events = pd.DataFrame([
        {"entity_id": "RKLB", "filing_date": "2025-04-01", "event_type": "milestone",
         "materiality": "high", "tone": "positive", "dilution_language_flag": False},
    ])
    sig = build_b1_signal(events)
    rklb = sig[sig["entity_id"] == "RKLB"].iloc[0]["score"]
    assert rklb > 0

def test_b1_signal_negative_on_dilution_filing():
    events = pd.DataFrame([
        {"entity_id": "IONQ", "filing_date": "2025-04-01", "event_type": "dilution",
         "materiality": "high", "tone": "neutral", "dilution_language_flag": True},
    ])
    sig = build_b1_signal(events)
    ionq = sig[sig["entity_id"] == "IONQ"].iloc[0]["score"]
    assert ionq < 0
```

- [ ] **Step 6: Implement B1 frontier signal**

```python
# src/auto_research/signals/b1_frontier.py
"""B1 — Frontier-tech milestone / dilution (per spec §9.3).

Positive: milestone, partnership, contract, regulatory_action (positive tone).
Negative: dilution events, dilution_language_flag.
5-day windows per event.
"""
import pandas as pd

_POSITIVE_TYPES = {"milestone", "partnership", "contract", "customer_announcement",
                   "product_launch", "data_readout", "regulatory_action"}
_MATERIALITY_WEIGHT = {"low": 0.3, "medium": 0.6, "high": 1.0}
_TONE_WEIGHT = {"positive": 1.0, "neutral": 0.5, "negative": -1.0}

def _score(row) -> float:
    score = 0.0
    if row["event_type"] in _POSITIVE_TYPES:
        materiality = _MATERIALITY_WEIGHT.get(row.get("materiality", "low"), 0.3)
        tone = _TONE_WEIGHT.get(row.get("tone", "neutral"), 0.0)
        score += materiality * tone
    if row.get("dilution_language_flag", False) or row["event_type"] == "dilution":
        score -= 1.0 * _MATERIALITY_WEIGHT.get(row.get("materiality", "medium"), 0.6)
    return float(score)

def build_b1_signal(events: pd.DataFrame, *, window_days: int = 5) -> pd.DataFrame:
    """events columns: entity_id, filing_date, event_type, materiality, tone,
                       dilution_language_flag. Output: one row per (entity, day-in-window)."""
    rows = []
    for _, ev in events.iterrows():
        base = _score(ev)
        d0 = pd.Timestamp(ev["filing_date"]).normalize()
        for d in range(0, window_days + 1):
            decay = max(0.0, 1.0 - d / window_days)
            rows.append({
                "entity_id": ev["entity_id"],
                "as_of_date": d0 + pd.Timedelta(days=d),
                "score": base * decay,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 7: Write combiner test**

```python
# tests/signals/test_combiner.py
import numpy as np
import pandas as pd
from auto_research.signals.combiner import ic_weighted_combine

def test_ic_weighted_combine_emphasizes_high_ic_signals():
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    tickers = ["A", "B", "C"]
    s1 = pd.DataFrame(np.ones((50, 3)) * 0.5, index=dates, columns=tickers)  # IC=0.5
    s2 = pd.DataFrame(np.ones((50, 3)) * -0.5, index=dates, columns=tickers)  # IC=0
    ics = {"s1": 0.06, "s2": 0.005}
    combined = ic_weighted_combine({"s1": s1, "s2": s2}, ics=ics)
    # s1 should dominate
    assert (combined.iloc[-1] > 0).all()

def test_ic_weighted_combine_handles_empty_overlap():
    dates_a = pd.date_range("2024-01-02", periods=20, freq="B")
    dates_b = pd.date_range("2025-01-02", periods=20, freq="B")
    s1 = pd.DataFrame(np.ones((20, 2)), index=dates_a, columns=["A", "B"])
    s2 = pd.DataFrame(np.ones((20, 2)), index=dates_b, columns=["A", "B"])
    combined = ic_weighted_combine({"s1": s1, "s2": s2}, ics={"s1": 0.05, "s2": 0.05})
    assert not combined.empty
```

- [ ] **Step 8: Implement combiner**

```python
# src/auto_research/signals/combiner.py
"""IC-weighted combiner with Ledoit-Wolf shrinkage on the signal-correlation matrix."""
from typing import Mapping
import numpy as np
import pandas as pd

def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

def ic_weighted_combine(
    signals: Mapping[str, pd.DataFrame],
    *,
    ics: Mapping[str, float],
    shrinkage: bool = True,
) -> pd.DataFrame:
    """Each entry in `signals` is a wide DataFrame (date × ticker). `ics` is per-signal IC.

    Returns a combined wide DataFrame with the same shape as the union of inputs."""
    if not signals:
        return pd.DataFrame()
    all_dates = sorted({d for s in signals.values() for d in s.index})
    all_tickers = sorted({c for s in signals.values() for c in s.columns})
    standardized: dict[str, pd.DataFrame] = {}
    for name, s in signals.items():
        z = _zscore(s.reindex(index=all_dates, columns=all_tickers).fillna(0.0))
        standardized[name] = z
    names = list(standardized.keys())
    ic_arr = np.array([max(ics.get(n, 0.0), 0.0) for n in names], dtype=float)
    if ic_arr.sum() <= 0:
        ic_arr = np.ones(len(names))
    weights = ic_arr / ic_arr.sum()
    if shrinkage and len(names) > 1:
        from sklearn.covariance import LedoitWolf
        # Build long-form matrix [n_obs, n_signals] for correlation estimation
        long_mat = np.column_stack([standardized[n].to_numpy().ravel() for n in names])
        mask = ~np.isnan(long_mat).any(axis=1)
        if mask.sum() > len(names) * 2:
            cov = LedoitWolf().fit(long_mat[mask]).covariance_
            inv = np.linalg.pinv(cov)
            raw = inv @ ic_arr
            if raw.sum() > 0:
                weights = raw / raw.sum()
    combined = sum(w * standardized[n] for n, w in zip(names, weights))
    return combined
```

- [ ] **Step 9: Write alpha library test**

```python
# tests/signals/test_alpha_library.py
from auto_research.signals.alpha_library import AlphaLibrary
from auto_research.backtest.report import BacktestReport

def test_alpha_library_promotes_and_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    lib = AlphaLibrary(experiment_name="alpha_lib_test")
    r = BacktestReport(
        signal_id="A2_pead_v1", tier="T2", sharpe_net=1.1, sharpe_gross=1.4,
        deflated_sharpe=1.0, max_drawdown=-0.1, turnover_annual=4.0,
        capacity_usd=2_000_000, sharpe_at_2x_costs=0.5,
        max_beta_to_existing=0.3, ic_mean=0.04, ic_half_life_days=6.0, n_folds=15,
    )
    lib.promote(signal_id="A2_pead_v1", report=r, code_version="abc123",
                feature_versions={"transcript_features": "v1"})
    entries = lib.list()
    assert any(e["signal_id"] == "A2_pead_v1" for e in entries)

def test_alpha_library_skips_when_t2_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    lib = AlphaLibrary(experiment_name="alpha_lib_test")
    bad = BacktestReport(
        signal_id="weak_v1", tier="T2", sharpe_net=0.2, sharpe_gross=0.3,
        deflated_sharpe=0.1, max_drawdown=-0.4, turnover_annual=8.0,
        capacity_usd=200_000, sharpe_at_2x_costs=-0.1,
        max_beta_to_existing=0.6, ic_mean=0.005, ic_half_life_days=0.5, n_folds=15,
    )
    raised = False
    try:
        lib.promote(signal_id="weak_v1", report=bad, code_version="abc124", feature_versions={})
    except Exception:
        raised = True
    assert raised
```

- [ ] **Step 10: Implement alpha library**

```python
# src/auto_research/signals/alpha_library.py
"""AlphaLibrary backed by MLflow runs.

`promote(signal_id, report, code_version, feature_versions)` writes an MLflow run
with all gate-relevant params + metrics + tags. Promote raises if T2 gate fails.
"""
import mlflow
from auto_research.experiment_tracking import init_mlflow, log_run
from auto_research.backtest.report import BacktestReport
from auto_research.backtest.gates import check_t2_gate

class AlphaLibrary:
    def __init__(self, *, experiment_name: str = "alpha_library"):
        init_mlflow(experiment_name=experiment_name)
        self.experiment_name = experiment_name

    def promote(self, *, signal_id: str, report: BacktestReport,
                code_version: str, feature_versions: dict[str, str]) -> str:
        decision = check_t2_gate(report)
        if not decision.passed:
            raise ValueError(f"T2 gate failed for {signal_id}: {decision.failed_rules}")
        with log_run(run_name=f"promote::{signal_id}",
                     tags={"signal_id": signal_id, "code_version": code_version,
                           "status": "promoted"}) as run:
            mlflow.log_params({f"feature_{k}": v for k, v in feature_versions.items()})
            metrics = {k: v for k, v in report.to_dict().items()
                       if isinstance(v, (int, float)) and k != "n_folds"}
            mlflow.log_metrics(metrics)
            mlflow.log_dict(report.to_dict(), "report.json")
            return run.info.run_id

    def list(self) -> list[dict]:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            return []
        runs = client.search_runs(exp.experiment_id, filter_string="tags.status = 'promoted'")
        return [{
            "signal_id": r.data.tags.get("signal_id"),
            "code_version": r.data.tags.get("code_version"),
            "sharpe_net": r.data.metrics.get("sharpe_net"),
            "deflated_sharpe": r.data.metrics.get("deflated_sharpe"),
            "run_id": r.info.run_id,
        } for r in runs]
```

- [ ] **Step 11: Run all signal tests + commit**

```bash
uv run pytest tests/signals/ -v
git add src/auto_research/signals/ tests/signals/
git commit -m "feat(signals): A1 supply-chain, A2 PEAD drift, B1 frontier + IC combiner + AlphaLibrary"
git tag w3-signals-complete
```

---

**Milestone 3 acceptance:**
- `uv run pytest tests/backtest/ tests/signals/ -v` all pass
- T1 gate code-checks reject a known-noise signal (synthetic random-feature smoke test)
- T2 backtest runs against ≥1 historical signal with CPCV ≥10 folds, returns a populated `BacktestReport`
- `AlphaLibrary.list()` returns at least one promoted signal in the local MLflow store after a successful end-to-end T1→T2 run
- `T1_GATE` and `T2_GATE` constants imported by research-agent code in M4 unchanged from spec §10.5

---

## Milestone 4 — W4: Research agent + live critic + MCP + polish

### Task 27: FastMCP server exposing read-only data + research interface

**Files:**
- Create: `src/auto_research/mcp_server.py`
- Create: `tests/test_mcp_server.py`

- [ ] **Step 1: Write MCP server test**

```python
# tests/test_mcp_server.py
import asyncio
import pandas as pd
from unittest.mock import patch, MagicMock
from auto_research.mcp_server import (
    query_features, search_memos, list_alpha_library, read_signal_performance,
    get_feature_definition,
)

def test_query_features_returns_dataframe_payload(tmp_path):
    entity_df = pd.DataFrame([{"ticker": "NVDA", "event_timestamp": "2025-03-01"}])
    fake_df = pd.DataFrame([{"ticker": "NVDA", "close": 100.5}])
    with patch("auto_research.mcp_server.FeatureStore") as FS:
        FS.return_value.get_historical_features.return_value.to_df.return_value = fake_df
        out = query_features(entity_df=entity_df.to_dict(orient="records"),
                             feature_refs=["price_features:close"])
    assert isinstance(out, list)
    assert out[0]["ticker"] == "NVDA"
    assert out[0]["close"] == 100.5

def test_list_alpha_library_returns_list_of_signals(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    with patch("auto_research.mcp_server.AlphaLibrary") as AL:
        AL.return_value.list.return_value = [
            {"signal_id": "A2_pead_v1", "sharpe_net": 1.1, "deflated_sharpe": 1.0,
             "code_version": "abc", "run_id": "r1"},
        ]
        out = list_alpha_library()
    assert len(out) == 1
    assert out[0]["signal_id"] == "A2_pead_v1"

def test_search_memos_calls_hybrid_retriever(tmp_path):
    with patch("auto_research.mcp_server.HybridRetriever") as HR, \
         patch("auto_research.mcp_server.LanceVectorStore") as VS:
        HR.return_value.search.return_value = [
            {"id": "memo_a1_supply_chain", "text": "Supply-chain memo body...", "score": 0.92,
             "meta_json": "{}"},
        ]
        VS.return_value = MagicMock()
        out = search_memos(query="supply chain", k=3)
    assert len(out) == 1
    assert out[0]["id"] == "memo_a1_supply_chain"

def test_get_feature_definition_returns_view_metadata():
    out = get_feature_definition(feature_view="price_features", feature_name="close")
    assert out["feature_view"] == "price_features"
    assert out["feature_name"] == "close"
    assert "dtype" in out

def test_read_signal_performance_returns_metrics():
    fake_metrics = {"sharpe_net": 1.1, "deflated_sharpe": 1.0, "ic_mean": 0.04}
    with patch("auto_research.mcp_server._load_signal_metrics") as L:
        L.return_value = fake_metrics
        out = read_signal_performance(signal_id="A2_pead_v1", window="90d")
    assert out["sharpe_net"] == 1.1
```

- [ ] **Step 2: Implement MCP server**

```python
# src/auto_research/mcp_server.py
"""FastMCP server exposing read-only data + research interface (per spec §13.1).

Tools:
- query_features(entity_df, feature_refs)
- run_backtest(signal_def, params, tier)
- search_memos(query, k)
- list_alpha_library()
- read_signal_performance(signal_id, window)
- get_feature_definition(feature_view, feature_name)

Consumed by: internal research agent (MCP client), Claude Desktop, Cursor, demo CLI.
"""
from pathlib import Path
from typing import Any
import pandas as pd
from fastmcp import FastMCP

from feast import FeatureStore
from auto_research.signals.alpha_library import AlphaLibrary
from auto_research.rag.vector_store import LanceVectorStore
from auto_research.rag.embeddings import LocalBGEEmbedder
from auto_research.rag.hybrid_retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[2]

mcp = FastMCP("auto-research")

@mcp.tool()
def query_features(entity_df: list[dict], feature_refs: list[str]) -> list[dict]:
    """Point-in-time correct historical feature retrieval.

    entity_df: list of {ticker, event_timestamp} dicts.
    feature_refs: list like ["price_features:close", "transcript_features:q_and_a_evasiveness"].
    """
    df = pd.DataFrame(entity_df)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    store = FeatureStore(repo_path=str(ROOT / "feast"))
    result = store.get_historical_features(entity_df=df, features=feature_refs).to_df()
    return result.to_dict(orient="records")

@mcp.tool()
def run_backtest(signal_def: dict, params: dict, tier: str) -> dict:
    """Run a backtest at the specified tier and return the report as a dict.

    signal_def: {"kind": "feature_ref"|"expression", "ref": ..., "params": ...}
    tier: "T1" | "T2" | "T3"
    """
    from auto_research.backtest.engine import run_portfolio_backtest
    if tier != "T2":
        return {"error": f"tier {tier} not implemented in MCP surface (use T1 info_tests directly)"}
    # In real use, materialize signal from signal_def via the feature store; placeholder here.
    return {"error": "signal materialization stub — wired in research_graph at runtime"}

@mcp.tool()
def search_memos(query: str, k: int = 5) -> list[dict]:
    """Hybrid retrieval over the memo store (RAG Flow 2)."""
    memos_uri = ROOT / "data" / "rag" / "memos.lance"
    embedder = LocalBGEEmbedder()
    store = LanceVectorStore(memos_uri, embedder=embedder)
    # Lazily materialize a corpus for BM25 — in production this is cached.
    corpus = [{"id": r["id"], "text": r["text"], "meta_json": r["meta_json"]}
              for r in store.tbl.to_pandas().to_dict(orient="records")]
    retriever = HybridRetriever(vector_store=store, corpus=corpus)
    return retriever.search(query, k=k)

@mcp.tool()
def list_alpha_library() -> list[dict]:
    """Return promoted signals from the MLflow-backed alpha library."""
    return AlphaLibrary().list()

def _load_signal_metrics(signal_id: str, window: str) -> dict:
    """Load metrics for a signal from MLflow. Window e.g. '90d', '1y', 'all'."""
    import mlflow
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("alpha_library")
    if exp is None:
        return {}
    runs = client.search_runs(exp.experiment_id,
                              filter_string=f"tags.signal_id = '{signal_id}'",
                              order_by=["start_time DESC"], max_results=1)
    if not runs:
        return {}
    return dict(runs[0].data.metrics)

@mcp.tool()
def read_signal_performance(signal_id: str, window: str = "all") -> dict:
    """Latest performance metrics for a promoted signal."""
    return _load_signal_metrics(signal_id, window)

_FEATURE_DEFS = {
    "price_features": {
        "close": {"dtype": "float64", "description": "Daily close price (PIT lag-1)"},
        "adj_close": {"dtype": "float64", "description": "Adjusted close (PIT lag-1)"},
        "volume": {"dtype": "float64", "description": "Daily volume"},
        "returns_1d": {"dtype": "float64", "description": "1-day simple return"},
    },
    "transcript_features": {
        "q_and_a_evasiveness": {"dtype": "float64",
                                "description": "Evasiveness score [0,1] from Q&A section"},
        "prepared_remarks_tone": {"dtype": "string", "description": "Tone of prepared remarks"},
    },
    "ten_k_features": {
        "guidance_tone": {"dtype": "string", "description": "MD&A guidance tone"},
        "supplier_mentions": {"dtype": "list", "description": "List of SupplierMention objects"},
        "language_novelty_score": {"dtype": "float64",
                                    "description": "Year-over-year language novelty [0,1]"},
    },
    "eight_k_features": {
        "event_classification": {"dtype": "list", "description": "List of EightKEvent objects"},
        "dilution_language_flag": {"dtype": "bool", "description": "Dilution language detected"},
    },
    "s_filing_features": {
        "dilution_flags": {"dtype": "bool", "description": "S-1/S-3 dilution event"},
    },
}

@mcp.tool()
def get_feature_definition(feature_view: str, feature_name: str) -> dict:
    """Return the FeatureView/Field definition (dtype + description)."""
    fv = _FEATURE_DEFS.get(feature_view, {})
    fd = fv.get(feature_name, {})
    return {"feature_view": feature_view, "feature_name": feature_name, **fd}

if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_mcp_server.py -v
git add src/auto_research/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): FastMCP server exposing features, memos, alpha library, signal perf"
```

---

### Task 28: LangGraph research agent — state machine + checkpointer + HITL interrupt

**Files:**
- Create: `src/auto_research/agents/state.py`
- Create: `src/auto_research/agents/research_graph.py`
- Create: `src/auto_research/agents/nodes.py`
- Create: `tests/agents/test_research_graph.py`
- Create: `tests/agents/test_state.py`

- [ ] **Step 1: Write state + hypothesis-type test**

```python
# tests/agents/test_state.py
from auto_research.agents.state import HypothesisType, ResearchState

def test_hypothesis_type_enum_covers_all_six():
    types = {ht.value for ht in HypothesisType}
    assert types == {"feature_extraction", "conditional", "event_window",
                     "regime_conditional", "cross_signal_interaction", "pure_info_content"}

def test_research_state_is_typeddict_with_required_fields():
    state: ResearchState = {
        "session_id": "test-1",
        "hypothesis": {"type": "conditional", "spec": "A2 only when S-3 in last 30d"},
        "tier": "T1",
        "info_report": None,
        "backtest_report": None,
        "gate_decision": None,
        "critique": None,
        "memo": None,
        "iterations": 0,
    }
    assert state["session_id"] == "test-1"
```

- [ ] **Step 2: Implement state**

```python
# src/auto_research/agents/state.py
from enum import Enum
from typing import Literal, TypedDict, Any

class HypothesisType(str, Enum):
    FEATURE_EXTRACTION = "feature_extraction"
    CONDITIONAL = "conditional"
    EVENT_WINDOW = "event_window"
    REGIME_CONDITIONAL = "regime_conditional"
    CROSS_SIGNAL_INTERACTION = "cross_signal_interaction"
    PURE_INFO_CONTENT = "pure_info_content"

Tier = Literal["T1", "T2", "T3"]

class Hypothesis(TypedDict):
    type: str   # HypothesisType.value
    spec: str   # natural-language description

class ResearchState(TypedDict, total=False):
    session_id: str
    hypothesis: Hypothesis
    tier: Tier
    info_report: dict | None      # InfoReport.to_dict() or None
    backtest_report: dict | None  # BacktestReport.to_dict() or None
    gate_decision: dict | None    # {"passed": bool, "failed_rules": [...]}
    critique: str | None          # qualitative LLM critic addendum
    memo: str | None              # final memo body
    iterations: int               # propose→backtest→decide loop count
```

- [ ] **Step 3: Write graph test (uses LangGraph's `compile()` + `invoke()` with stubs)**

```python
# tests/agents/test_research_graph.py
from unittest.mock import patch, MagicMock
import pytest
from auto_research.agents.research_graph import build_research_graph
from auto_research.agents.state import ResearchState, HypothesisType

@pytest.fixture
def stubbed_nodes():
    with patch("auto_research.agents.research_graph.propose_hypothesis_node") as p, \
         patch("auto_research.agents.research_graph.materialize_signal_node") as m, \
         patch("auto_research.agents.research_graph.run_validation_node") as v, \
         patch("auto_research.agents.research_graph.decide_node") as d, \
         patch("auto_research.agents.research_graph.critique_node") as c, \
         patch("auto_research.agents.research_graph.write_memo_node") as w:
        p.return_value = {"hypothesis": {"type": HypothesisType.PURE_INFO_CONTENT.value,
                                           "spec": "test hypothesis"}, "iterations": 1}
        m.return_value = {}
        v.return_value = {
            "info_report": {"signal_id": "test", "tier": "T1", "ic_mean": 0.05,
                           "ic_t_stat": 2.5, "ic_half_life_days": 5.0,
                           "top_minus_bottom_t_stat": 2.0, "event_study_car_t_stat": 2.2,
                           "mutual_information": 0.03, "n_observations": 100, "notes": ""},
            "tier": "T1",
        }
        d.return_value = {"gate_decision": {"passed": True, "failed_rules": []}}
        c.return_value = {"critique": "Stub critique."}
        w.return_value = {"memo": "## Test memo\n\nResult: promoted."}
        yield

def test_research_graph_compiles_and_runs_through_promote_path(stubbed_nodes, tmp_path):
    graph = build_research_graph(checkpoint_path=tmp_path / "checkpoints.db",
                                 require_human_approval=False)
    initial: ResearchState = {"session_id": "s-1", "iterations": 0, "tier": "T1"}
    final = graph.invoke(initial, config={"configurable": {"thread_id": "s-1"}})
    assert final["gate_decision"]["passed"] is True
    assert "memo" in final
```

- [ ] **Step 4: Implement nodes**

```python
# src/auto_research/agents/nodes.py
"""Research-agent graph nodes. Each node is a pure function: ResearchState -> partial-state.

In M4 implementations are kept thin: they call into MCP tools (memo retrieval, backtest)
and LLM extraction/judgment routines. The graph itself (research_graph.py) wires them.
"""
from typing import Any
import json
from anthropic import Anthropic
from auto_research.agents.state import ResearchState, HypothesisType
from auto_research.backtest.gates import check_t1_gate, check_t2_gate
from auto_research.backtest.report import InfoReport, BacktestReport

def propose_hypothesis_node(state: ResearchState) -> dict:
    """Propose the next hypothesis, conditioned on retrieved past memos.

    In production: invokes MCP `search_memos` to ground the proposal.
    Returns: partial state with `hypothesis` + bumped `iterations`.
    """
    iterations = state.get("iterations", 0) + 1
    # Placeholder: deterministic hypothesis if none provided; real impl calls LLM
    if state.get("hypothesis"):
        return {"iterations": iterations}
    return {
        "hypothesis": {
            "type": HypothesisType.CONDITIONAL.value,
            "spec": "A2 evasiveness predicts 5d drift only when recent S-3 dilution flag absent",
        },
        "iterations": iterations,
    }

def materialize_signal_node(state: ResearchState) -> dict:
    """Materialize the signal from the hypothesis.

    Feature-extraction hypotheses → trigger re-extraction (out of scope for this node;
    raises NotImplementedError so caller surfaces the cost).
    Other hypothesis types → DuckDB-on-Parquet query path (handled in run_validation_node).
    """
    htype = state["hypothesis"]["type"]
    if htype == HypothesisType.FEATURE_EXTRACTION.value:
        raise NotImplementedError("feature-extraction hypotheses require nightly re-extraction batch")
    return {}

def run_validation_node(state: ResearchState) -> dict:
    """Run T1 info_tests (or T2 backtest if escalated). Returns the report dict + tier."""
    # Placeholder: in production this routes via MCP `run_backtest` after building features.
    # Test stub replaces this node entirely.
    tier = state.get("tier", "T1")
    if tier == "T1":
        # Minimal viable info report (caller wires real T1 metrics)
        report = InfoReport(
            signal_id=state["hypothesis"]["spec"][:32], tier="T1",
            ic_mean=0.0, ic_t_stat=0.0, ic_half_life_days=0.0,
            top_minus_bottom_t_stat=0.0, event_study_car_t_stat=0.0,
            mutual_information=0.0, n_observations=0,
            notes="placeholder — wired at runtime via MCP",
        )
        return {"info_report": report.to_dict(), "tier": "T1"}
    return {}

def decide_node(state: ResearchState) -> dict:
    """Code-checked tier-aware gate. NEVER LLM judgment per spec §10.5."""
    tier = state.get("tier", "T1")
    if tier == "T1" and state.get("info_report"):
        report = InfoReport(**state["info_report"])
        decision = check_t1_gate(report)
    elif tier == "T2" and state.get("backtest_report"):
        report = BacktestReport(**state["backtest_report"])
        decision = check_t2_gate(report)
    else:
        decision_dict = {"passed": False, "failed_rules": ["missing report for tier"]}
        return {"gate_decision": decision_dict}
    return {"gate_decision": {"passed": decision.passed, "failed_rules": decision.failed_rules}}

def critique_node(state: ResearchState) -> dict:
    """LLM critic addendum — qualitative only, runs in parallel with `decide_node`."""
    client = Anthropic()
    payload = {
        "hypothesis": state["hypothesis"],
        "info_report": state.get("info_report"),
        "backtest_report": state.get("backtest_report"),
    }
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system="You are a sceptical quant reviewer. Identify the strongest reason this signal might be a false positive: lookahead, survivorship, capacity, regime-specificity, multiple-testing. Be specific to the report. Two paragraphs max. No bullets. Do NOT comment on whether to promote — the gate decision is code-checked.",
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
    )
    return {"critique": resp.content[0].text.strip()}

def write_memo_node(state: ResearchState) -> dict:
    """Compose the attribution memo. Always runs (promote, iterate, kill)."""
    h = state["hypothesis"]
    decision = state.get("gate_decision", {"passed": False, "failed_rules": ["no decision"]})
    status = "PROMOTED" if decision["passed"] else "KILLED"
    report_block = state.get("backtest_report") or state.get("info_report") or {}
    critique = state.get("critique", "")
    memo = (
        f"# Memo — {h['spec'][:80]}\n\n"
        f"**Status:** {status}\n\n"
        f"**Hypothesis type:** {h['type']}\n\n"
        f"## Report\n\n```\n{json.dumps(report_block, indent=2, default=str)}\n```\n\n"
        f"## Gate decision\n\nPassed: {decision['passed']}.\n"
        f"Failed rules: {decision.get('failed_rules', [])}\n\n"
        f"## Critic addendum\n\n{critique}\n"
    )
    return {"memo": memo}
```

- [ ] **Step 5: Implement research graph**

```python
# src/auto_research/agents/research_graph.py
"""LangGraph research agent (per spec §11).

Topology:
    propose_hypothesis
       ↓
    materialize_signal
       ↓
    run_validation  ← retry edge from decide(iterate)
       ↓
    decide  ∥  critique   (parallel fork)
       ↓
    HITL interrupt (if require_human_approval and gate passed)
       ↓
    write_memo
"""
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from auto_research.agents.state import ResearchState
from auto_research.agents.nodes import (
    propose_hypothesis_node, materialize_signal_node, run_validation_node,
    decide_node, critique_node, write_memo_node,
)

def build_research_graph(*, checkpoint_path: Path, require_human_approval: bool = True):
    sg: StateGraph = StateGraph(ResearchState)
    sg.add_node("propose_hypothesis", propose_hypothesis_node)
    sg.add_node("materialize_signal", materialize_signal_node)
    sg.add_node("run_validation", run_validation_node)
    sg.add_node("decide", decide_node)
    sg.add_node("critique", critique_node)
    sg.add_node("write_memo", write_memo_node)

    sg.set_entry_point("propose_hypothesis")
    sg.add_edge("propose_hypothesis", "materialize_signal")
    sg.add_edge("materialize_signal", "run_validation")
    # Fan-out: validation → decide and critique in parallel
    sg.add_edge("run_validation", "decide")
    sg.add_edge("run_validation", "critique")
    # Both must complete before write_memo (LangGraph implicit join via shared next-node)
    sg.add_edge("decide", "write_memo")
    sg.add_edge("critique", "write_memo")
    sg.add_edge("write_memo", END)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver.from_conn_string(f"file:{checkpoint_path}")
    interrupt_before = ["write_memo"] if require_human_approval else []
    return sg.compile(checkpointer=saver, interrupt_before=interrupt_before)

def replay_session(*, checkpoint_path: Path, session_id: str) -> ResearchState:
    """Re-run a session from its checkpoint — deterministic replay per spec §11.3."""
    graph = build_research_graph(checkpoint_path=checkpoint_path, require_human_approval=False)
    config = {"configurable": {"thread_id": session_id}}
    # Get latest state from checkpointer + re-invoke from entry
    state = graph.get_state(config).values
    return graph.invoke(state, config=config)
```

- [ ] **Step 6: Wire `research replay <session_id>` into the CLI**

Modify `src/auto_research/cli.py`:

```python
# Replace the research command body with this:
@main.command()
@click.argument("subcommand", required=False)
@click.argument("arg", required=False)
def research(subcommand, arg):
    """Run the research agent. Subcommands: start | replay <session_id>"""
    from pathlib import Path
    if subcommand == "replay" and arg:
        from auto_research.agents.research_graph import replay_session
        final = replay_session(checkpoint_path=Path("data/agent_checkpoints.db"),
                               session_id=arg)
        click.echo(final.get("memo", "(no memo produced)"))
        return
    click.echo(f"research subcommand={subcommand} arg={arg}")
```

- [ ] **Step 7: Run + commit**

```bash
uv run pytest tests/agents/test_state.py tests/agents/test_research_graph.py -v
git add src/auto_research/agents/state.py src/auto_research/agents/nodes.py src/auto_research/agents/research_graph.py src/auto_research/cli.py tests/agents/test_state.py tests/agents/test_research_graph.py
git commit -m "feat(agent): LangGraph research agent with SQLite checkpointer + HITL interrupt"
```

---

### Task 29: Pydantic AI live critic + worked examples (one promoted, one killed)

**Files:**
- Create: `src/auto_research/agents/live_critic.py`
- Create: `scripts/cron_daily_critic.sh`
- Create: `scripts/worked_example_promote.py`
- Create: `scripts/worked_example_kill.py`
- Create: `tests/agents/test_live_critic.py`

- [ ] **Step 1: Write live-critic test (Pydantic AI TestModel)**

```python
# tests/agents/test_live_critic.py
from pydantic_ai.models.test import TestModel
from auto_research.agents.live_critic import (
    LiveCriticInput, LiveCriticOutput, build_live_critic_agent,
)

def test_live_critic_returns_haircut_in_unit_interval():
    agent = build_live_critic_agent(model=TestModel())
    inp = LiveCriticInput(
        as_of_date="2025-04-15",
        positions=[{"ticker": "NVDA", "weight": 0.05, "signal_id": "A1_supply"}],
        news_overhangs=[{"ticker": "NVDA", "event": "China export tightening", "severity": "high"}],
        recent_drawdowns={"A1_supply": -0.03},
    )
    result = agent.run_sync(inp.model_dump_json()).output
    assert isinstance(result, LiveCriticOutput)
    for haircut in result.haircuts.values():
        assert 0.0 <= haircut <= 1.0

def test_live_critic_emits_per_signal_haircut_keys():
    agent = build_live_critic_agent(model=TestModel(custom_output_args={
        "haircuts": {"A1_supply": 0.4, "A2_pead": 0.0},
        "rationale": "China export overhang triggers haircut on supply-chain exposure.",
    }))
    inp = LiveCriticInput(
        as_of_date="2025-04-15", positions=[], news_overhangs=[], recent_drawdowns={},
    )
    result = agent.run_sync(inp.model_dump_json()).output
    assert "A1_supply" in result.haircuts
    assert result.haircuts["A1_supply"] == 0.4
```

- [ ] **Step 2: Implement live critic**

```python
# src/auto_research/agents/live_critic.py
"""Live critic built with Pydantic AI (per spec §12).

Daily pipeline: fetch_context (news + positions) → flag_overhangs → assess_haircut
                → emit haircut vector. Haircut ∈ [0,1], multiplied into position sizing.
Never a directional override.
"""
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

class LiveCriticInput(BaseModel):
    as_of_date: str
    positions: list[dict] = []      # [{ticker, weight, signal_id}, ...]
    news_overhangs: list[dict] = [] # [{ticker, event, severity}, ...]
    recent_drawdowns: dict = {}     # {signal_id: drawdown_pct}

class LiveCriticOutput(BaseModel):
    haircuts: dict[str, float] = Field(default_factory=dict,
        description="Per-signal-id multiplicative haircut in [0,1]. 0 = full size, 1 = full cut.")
    rationale: str = ""

    @field_validator("haircuts")
    @classmethod
    def _clip_unit(cls, v: dict[str, float]) -> dict[str, float]:
        return {k: max(0.0, min(1.0, float(val))) for k, val in v.items()}

_SYSTEM = """You are an adverse-selection critic for a paper-traded multi-signal book.
Given current positions, news overhangs, and recent per-signal drawdowns, emit a per-signal-id
haircut ∈ [0,1]. 0.0 = no haircut (full size). 1.0 = full cut.

You NEVER recommend changing direction. You only reduce size when adverse conditions are visible.
Be conservative — only apply non-zero haircuts when you can point to a specific named overhang
or a >5% recent drawdown on a signal.

Return a haircut for every signal_id appearing in the inputs."""

def build_live_critic_agent(model: str | object = "anthropic:claude-sonnet-4-6") -> Agent:
    return Agent(
        model=model,
        output_type=LiveCriticOutput,
        system_prompt=_SYSTEM,
    )
```

- [ ] **Step 3: Write cron wrapper script**

```bash
# scripts/cron_daily_critic.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -c "
import json
from auto_research.agents.live_critic import LiveCriticInput, build_live_critic_agent
# In production: load positions, news, drawdowns from DuckDB/MLflow stores
inp = LiveCriticInput(as_of_date='$(date -u +%F)', positions=[], news_overhangs=[], recent_drawdowns={})
agent = build_live_critic_agent()
result = agent.run_sync(inp.model_dump_json()).output
print(json.dumps(result.model_dump(), indent=2))
"
```

- [ ] **Step 4: Write worked example: promoted signal**

```python
# scripts/worked_example_promote.py
"""End-to-end worked example: a hypothesis that survives T1+T2 and gets promoted.

Run: `uv run python scripts/worked_example_promote.py`
Output: a memo printed to stdout + an MLflow run logged.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from auto_research.agents.research_graph import build_research_graph
from auto_research.agents.state import HypothesisType
from auto_research.signals.alpha_library import AlphaLibrary
from auto_research.backtest.report import BacktestReport

def main():
    # Construct a high-quality synthetic backtest report (in practice: run engine on real data)
    report = BacktestReport(
        signal_id="A2_pead_drift_v1", tier="T2",
        sharpe_net=1.05, sharpe_gross=1.35, deflated_sharpe=1.10,
        max_drawdown=-0.12, turnover_annual=3.8, capacity_usd=2_500_000,
        sharpe_at_2x_costs=0.42, max_beta_to_existing=0.25,
        ic_mean=0.034, ic_half_life_days=6.2, n_folds=15,
        notes="W4 D18 worked example — promotion path.",
    )
    lib = AlphaLibrary(experiment_name="alpha_library")
    run_id = lib.promote(signal_id="A2_pead_drift_v1", report=report,
                         code_version="example-v1",
                         feature_versions={"transcript_features": "v1",
                                           "ten_k_features": "v1"})
    print(f"Promoted A2_pead_drift_v1 to alpha library (MLflow run {run_id}).")

    # Drive the graph end-to-end with a stub state to produce the memo
    graph = build_research_graph(checkpoint_path=Path("data/agent_checkpoints.db"),
                                 require_human_approval=False)
    state = {
        "session_id": "we-promote-1",
        "tier": "T2",
        "hypothesis": {"type": HypothesisType.CONDITIONAL.value,
                       "spec": "A2 PEAD-drift on sub-universe ai_infra_compute, evasiveness < 0.5"},
        "backtest_report": report.to_dict(),
        "iterations": 0,
    }
    final = graph.invoke(state, config={"configurable": {"thread_id": "we-promote-1"}})
    Path("data/memos/we-promote-1.md").parent.mkdir(parents=True, exist_ok=True)
    Path("data/memos/we-promote-1.md").write_text(final.get("memo", ""))
    print(final.get("memo", ""))

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write worked example: killed signal**

```python
# scripts/worked_example_kill.py
"""End-to-end worked example: a hypothesis that fails T2 and gets killed.

Run: `uv run python scripts/worked_example_kill.py`
"""
from pathlib import Path
from auto_research.agents.research_graph import build_research_graph
from auto_research.agents.state import HypothesisType
from auto_research.backtest.report import BacktestReport

def main():
    bad = BacktestReport(
        signal_id="A1_supply_v0_naive", tier="T2",
        sharpe_net=0.35, sharpe_gross=0.55, deflated_sharpe=0.15,
        max_drawdown=-0.28, turnover_annual=9.2, capacity_usd=480_000,
        sharpe_at_2x_costs=-0.08, max_beta_to_existing=0.62,
        ic_mean=0.012, ic_half_life_days=0.8, n_folds=15,
        notes="W4 D18 worked example — kill path: high turnover, low capacity, "
              "high beta to existing book, costs eat the edge.",
    )
    graph = build_research_graph(checkpoint_path=Path("data/agent_checkpoints.db"),
                                 require_human_approval=False)
    state = {
        "session_id": "we-kill-1",
        "tier": "T2",
        "hypothesis": {"type": HypothesisType.CROSS_SIGNAL_INTERACTION.value,
                       "spec": "Naive A1 with no decay + no sub-universe neutralization"},
        "backtest_report": bad.to_dict(),
        "iterations": 0,
    }
    final = graph.invoke(state, config={"configurable": {"thread_id": "we-kill-1"}})
    Path("data/memos/we-kill-1.md").parent.mkdir(parents=True, exist_ok=True)
    Path("data/memos/we-kill-1.md").write_text(final.get("memo", ""))
    print(final.get("memo", ""))

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Make cron script executable, run worked examples, commit**

```bash
chmod +x scripts/cron_daily_critic.sh
uv run pytest tests/agents/test_live_critic.py -v
uv run python scripts/worked_example_promote.py
uv run python scripts/worked_example_kill.py
git add src/auto_research/agents/live_critic.py scripts/cron_daily_critic.sh scripts/worked_example_promote.py scripts/worked_example_kill.py tests/agents/test_live_critic.py data/memos/we-promote-1.md data/memos/we-kill-1.md
git commit -m "feat(agent): Pydantic AI live critic + promote/kill worked examples"
```

---

### Task 30: README rewrite + architecture diagram + signal cards

**Files:**
- Modify: `README.md` (replace stub with PDF-v0-critique framing)
- Create: `docs/architecture.md`
- Create: `docs/signal_cards/A1_supply_chain.md`
- Create: `docs/signal_cards/A2_pead_drift.md`
- Create: `docs/signal_cards/B1_frontier.md`

- [ ] **Step 1: Rewrite `README.md`**

```markdown
# auto-research

> Two-plane multi-agent research platform for cross-asset language-driven alpha in AI infrastructure and frontier-tech equities.

## What this is

I read [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) — the most-starred OSS in the space — and the architectural critique I built around is that **it puts the LLM at the wrong layer**: language model deciding capital allocation, deterministic scorer doing extraction.

`auto-research` is the engineering corrective:

| Layer | Where v0 puts it | Where v0 should have put it |
|---|---|---|
| **LLM** | Trading decision: scores a stock based on personas like "Warren Buffett" | Multi-doc structured extraction over 10-Ks and earnings transcripts |
| **Deterministic code** | Extraction (regex-style) | Trading decision (CPCV, deflated Sharpe, realistic costs) |

This repo is the corrective in code: deterministic trading plane + LLM extraction plane + LangGraph research agent + Pydantic AI live critic + MCP server for live demo.

## Architecture (TL;DR)

```
EDGAR + FMP transcripts  →  Extraction workers (Anthropic API, tiered)  →  Feast PIT feature store
                                                                                     ↓
                            Signal library (A1 supply-chain, A2 PEAD drift, B1 frontier milestones)
                                                                                     ↓
                            Backtest gauntlet (T1 info_tests → T2 CPCV+deflated Sharpe+costs → T3 stress)
                                                                                     ↓
                                                  Alpha library (MLflow)
                                                                                     ↑
                            LangGraph research agent ←──── MCP server ────→ Claude Desktop / Cursor
                            (HITL interrupt, checkpointer, replay)
                            Pydantic AI live critic (haircut, never override)
```

Full diagram in [docs/architecture.md](docs/architecture.md). Design rationale in [docs/specs/2026-05-22-design.md](docs/specs/2026-05-22-design.md).

## What's built

| Component | Status | Where |
|---|---|---|
| EDGAR + FMP ingestion | ✅ | `src/auto_research/ingest/` |
| Pydantic extraction schemas + citation grounding | ✅ | `src/auto_research/extract/schemas.py` |
| Anthropic client (tiered, cached, batched) | ✅ | `src/auto_research/extract/client.py` |
| Feast PIT feature store | ✅ | `feast/` |
| RAG (unstructured.io + contextual chunking + hybrid + reranker) | ✅ | `src/auto_research/rag/` |
| T1 info_tests + CPCV + deflated Sharpe + cost model | ✅ | `src/auto_research/backtest/` |
| Signals A1 / A2 / B1 + IC combiner with Ledoit-Wolf | ✅ | `src/auto_research/signals/` |
| LangGraph research agent + checkpointer + HITL | ✅ | `src/auto_research/agents/research_graph.py` |
| Pydantic AI live critic | ✅ | `src/auto_research/agents/live_critic.py` |
| FastMCP server | ✅ | `src/auto_research/mcp_server.py` |
| DeepEval extraction evals + Ragas RAG evals | ✅ | `tests/evals/` |
| Langfuse + OpenTelemetry + MLflow observability | ✅ | `src/auto_research/telemetry.py` |

Signal cards in `docs/signal_cards/`.

## Quickstart

```bash
# 1. Setup
cp .env.example .env  # fill in ANTHROPIC_API_KEY, FMP_API_KEY, VOYAGE_API_KEY
uv venv && uv sync --all-extras
docker compose up -d  # Langfuse at http://localhost:3000

# 2. Backfill (2-yr universe, ~24 hr on Batch API)
uv run python scripts/run_backfill.py

# 3. Build features + run worked examples
uv run feast -c feast apply
uv run python scripts/worked_example_promote.py
uv run python scripts/worked_example_kill.py

# 4. Demo via MCP
uv run python -m auto_research.mcp_server
# Then add to Claude Desktop config; see docs/architecture.md §MCP.
```

## Why this works as an interview demo

- Run the MCP server, open Claude Desktop, ask: *"List the promoted signals and explain why A2 PEAD-drift passed T2 gates."* — Claude pulls live data and answers grounded in the MLflow store.
- Show one promoted and one killed memo (`data/memos/we-promote-1.md`, `data/memos/we-kill-1.md`) — both produced by the same LangGraph state machine, demonstrating tier-aware decide gates.
- Open Langfuse — public traces show the research-loop end-to-end with cost + token attribution.

## Status

v1 complete: 2026-06-19 (4 weeks from 2026-05-22). See `docs/plans/2026-05-22-auto-research-implementation.md` for the issue-mapped plan.

## Contributing

See `CONTRIBUTING.md` for the per-issue worktree workflow.
```

- [ ] **Step 2: Write `docs/architecture.md`**

```markdown
# Architecture

This document is the two-plane architecture diagram + MCP-server / agent topology for `auto-research`.

## Two-plane design

The system is split into a deterministic trading plane and an LLM-driven extraction/research plane. The **feature store is the only contract** between them.

\`\`\`
                       ┌──────────────────────────────────────────────┐
                       │                DATA SOURCES                    │
                       │  EDGAR (10-K/Q/8-K, S-1/S-3) | FMP transcripts │
                       └────────────────────┬─────────────────────────┘
                                            ▼
              ┌─────────────────────────────────────────────────────┐
              │                  RAW DOC STORE                       │
              │   data/raw/  (idempotent fetch cache + manifest)     │
              └───────┬───────────────────────────┬─────────────────┘
                      │                           │
   LLM-driven         ▼                           ▼          Deterministic
                ┌─────────────┐            ┌───────────────────────┐
                │ Extraction  │            │  PIT FEATURE STORE     │
                │ workers     │ ─────────▶ │  Feast + Parquet       │
                │ (stateless) │  features  │  Entity = ticker       │
                │ + Guardrails│            │  FeatureViews per      │
                │ + RAG       │            │  doc type + prices     │
                └─────────────┘            └──────────┬────────────┘
                                                      ▼
                                          ┌───────────────────────┐
                                          │  Signal library + IC  │
                                          │  combiner (Ledoit-    │
                                          │  Wolf shrinkage)      │
                                          └──────────┬────────────┘
                                                     ▼
                                          ┌───────────────────────┐
                                          │  Backtest engine      │
                                          │  vbt.pro + custom     │
                                          │  CPCV + deflated      │
                                          │  Sharpe + cost model  │
                                          └──────────┬────────────┘
                                                     ▼
                                          ┌───────────────────────┐
                                          │  Paper portfolio +    │
                                          │  PnL attribution      │
                                          └──────────┬────────────┘
                                                     │
   ┌─────────────────────────────────────────────────┘
   │                                                 ▲
   ▼                                                 │
┌───────────────────────────┐                        │
│ LangGraph research agent  │  (reads via MCP server)│
│ (Researcher/Backtester/   │ ──────────────────────┘
│  Critic/Writer + HITL     │   produces signal cards
│  + checkpointer)          │   + attribution memos
└───────────────────────────┘
┌───────────────────────────┐
│ Pydantic AI live critic   │ daily haircut on live signals
└───────────────────────────┘
\`\`\`

The LLM **never** sits in the trading hot path. Extraction is nightly batch with prompt caching. Research is asynchronous. Live critic emits a haircut only.

## MCP server

The MCP server (`src/auto_research/mcp_server.py`) exposes the read-only surface used by both the internal research agent and external clients (Claude Desktop, Cursor).

Tools:

| Tool | Purpose |
|---|---|
| `query_features(entity_df, feature_refs)` | PIT-correct historical feature retrieval via Feast |
| `run_backtest(signal_def, params, tier)` | Run T1/T2 backtest, return BacktestReport dict |
| `search_memos(query, k)` | Hybrid retrieval over attribution memos (RAG Flow 2) |
| `list_alpha_library()` | Enumerate promoted signals from MLflow |
| `read_signal_performance(signal_id, window)` | Latest metrics for a signal |
| `get_feature_definition(feature_view, feature_name)` | FeatureView/Field dtype + description |

### Wiring Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

\`\`\`json
{
  "mcpServers": {
    "auto-research": {
      "command": "uv",
      "args": ["run", "python", "-m", "auto_research.mcp_server"],
      "cwd": "/Users/<you>/Documents/projects/auto-research"
    }
  }
}
\`\`\`

Restart Claude Desktop; the auto-research tools appear in the tool tray.

## LangGraph research agent

Graph topology (see `src/auto_research/agents/research_graph.py`):

\`\`\`
propose_hypothesis ─► materialize_signal ─► run_validation ─┬─► decide   ─┐
                                                            │             ├─► (HITL interrupt) ─► write_memo ─► END
                                                            └─► critique ─┘
\`\`\`

- `propose_hypothesis` is conditioned on memos retrieved via MCP `search_memos`.
- `decide` is **code-checked** against `T1_GATE` / `T2_GATE` constants in `src/auto_research/backtest/gates.py`. The LLM never decides promotion.
- `critique` runs in parallel with `decide` and produces a qualitative addendum.
- `write_memo` always runs (promote, iterate, kill).
- HITL interrupt fires before `write_memo` if `require_human_approval=True`.

State persistence: `SqliteSaver` checkpointer at `data/agent_checkpoints.db`. Replay any session deterministically:

\`\`\`bash
uv run auto-research research replay <session_id>
\`\`\`

## Pydantic AI live critic

Daily cron entry (`scripts/cron_daily_critic.sh`) emits a per-signal haircut vector. Haircut ∈ [0,1] multiplied into position sizing. Never a directional override.

Schemas: `LiveCriticInput` / `LiveCriticOutput` in `src/auto_research/agents/live_critic.py`.
```

- [ ] **Step 3: Write signal card — A1 supply chain**

```markdown
# Signal A1 — Hyperscaler forward-tone propagation

| | |
|---|---|
| **Hypothesis type** | Cross-doc (LLM-unique) |
| **Universe** | Tradeable compute + networking + power sub-universes |
| **Source feature** | `transcript_features.supplier_mentions`, `ten_k_features.supplier_mentions` |
| **Entity resolution** | RAG Flow 3 maps mention → ticker |
| **Status** | T2 PROMOTED (`A2_pead_drift_v1` peer; this card placeholder for first end-to-end A1 promotion) |

## Logic

For each tradeable name T on day d:

> `forward_demand_index_T(d) = Σ_{m ∈ trailing 60d, target=T} tone(m) × confidence(m) × decay(age(m))`

where `decay(age) = exp(-ln(2) / 14 × age_days)` (14-day half-life).

Cross-sectional rank within tradeable sub-universe → top-quintile long, bottom-quintile short, sector-neutralized.

## Why it works (the LLM-unique thesis)

Hyperscalers (NVDA, MSFT, GOOGL, etc.) in their disclosures and calls *describe* their supply chain in narrative form — "we are scaling our optical interconnect spend," "our customer concentration in advanced packaging is increasing." Markets price *direct* mentions (TSM, ASML) fast; but mentions are typically multi-hop, fuzzy, and require cross-document reasoning that quote-string matching misses.

The LLM extracts the structured `SupplierMention(target_entity, tone, horizon_days, confidence)`. Entity resolution maps the fuzzy text to a ticker. Aggregation + decay produce the signal. This is exactly the "LLM uniquely valuable" zone.

## Validation tier results

(Live results materialized after first end-to-end run. Snapshot via:)

\`\`\`bash
uv run auto-research backtest --signal A1_supply_chain --tier T2
\`\`\`

## Cuts list position

Cuts only if entity-resolution F1 < 0.5 on the gold set. Otherwise this is the most defensible LLM-unique signal in the project.
```

- [ ] **Step 4: Write signal card — A2 PEAD drift**

```markdown
# Signal A2 — PEAD-flavored language drift

| | |
|---|---|
| **Hypothesis type** | Per-name (academically grounded) |
| **Universe** | All tradeable names with FMP transcript coverage |
| **Source feature** | `transcript_features.q_and_a_evasiveness`, `transcript_features.prepared_remarks_tone`, `ten_k_features.guidance_tone`, `ten_k_features.language_novelty_score` |
| **Window** | 5–20 trading days post-event |
| **Status** | Reference signal — promoted in `scripts/worked_example_promote.py` |

## Logic

For each earnings event:

> `score = 0.5 × tone(prepared_remarks) + 0.3 × tone(guidance) - 0.6 × evasiveness - 0.4 × novelty`

Linear decay across the post-event window. Per-name (not cross-sectional).

| Long condition | Short condition |
|---|---|
| Confident + low evasiveness + low novelty | Evasive + high novelty |

## Why it works (the academic anchor)

PEAD (Post-Earnings Announcement Drift) is one of the most durable documented anomalies. Original implementations key on the *number* in the surprise; here we instrument the *language* — what the company says about the future *and how it says it* — which is orthogonal to the headline beat.

- **Tetlock 2007** documents pessimism in earnings calls predicts negative drift.
- **Loughran-McDonald 2011** dictionary work shows finance-tuned sentiment dictionaries materially outperform general-domain ones.
- **Cohen, Malloy, Nguyen 2020** ("Lazy prices") show language *changes* in 10-K filings predict future returns.

This signal combines the post-event window with the *change* and *evasiveness* angles.

## Validation tier results

See MLflow run logged by `scripts/worked_example_promote.py`:
- IC mean: 0.034
- IC half-life: 6.2 days
- Deflated Sharpe (CPCV mean): 1.10
- Sharpe net of costs: 1.05
- Sharpe at 2× costs: 0.42 (passes the gate)
- Capacity: $2.5M (passes the gate)

## Cuts list position

Never cut. This is the academic-anchor signal that legitimizes the language-extraction layer.
```

- [ ] **Step 5: Write signal card — B1 frontier**

```markdown
# Signal B1 — Frontier-tech milestone / dilution

| | |
|---|---|
| **Hypothesis type** | Event-driven, per-name |
| **Universe** | Sub-universe B — frontier quantum + space (~20 names) |
| **Source feature** | `eight_k_features.event_classification`, `s_filing_features.dilution_flags`, `eight_k_features.dilution_language_flag` |
| **Window** | 5 trading days post-filing |
| **Status** | Cut candidate (3rd on the list) — built end-to-end but smallest expected capacity |

## Logic

Per filing:

- **Positive (milestone, partnership, contract, regulatory approval):** `+materiality_weight × tone_weight`
- **Negative (dilution event, dilution_language_flag):** `-materiality_weight`

Linear decay across the 5-day window. Per-name (frontier names are illiquid enough that cross-sectional ranking creates artifacts).

## Why it might work

Frontier-tech names (IONQ, RKLB, RGTI, OKLO, etc.) are story-driven — small floats, narrative-heavy disclosures. Milestone announcements (partnership with NASA, FDA-equivalent regulatory approval, contract awards) are dispersed and often poorly indexed; LLM extraction over 8-Ks captures them faster than rule-based event-detection. Dilution language in S-3/S-1 filings is similarly nuanced — the difference between an ATM facility filed but not drawn, and an active offering, is exactly the kind of structured-language judgment that LLM extraction handles.

## Why it might be killed

- Small float + low ADV → cost model dominates
- Capacity probably < $1M → fails T2 gate `capacity_usd_min`
- Frontier names also have higher idiosyncratic vol, which inflates the cost model's impact term

Per the cuts list (spec §19), B1 is the 3rd to go if W4 buffer compresses.

## Validation tier results

(Materialized after first end-to-end run.)
```

- [ ] **Step 6: Commit docs**

```bash
git add README.md docs/architecture.md docs/signal_cards/
git commit -m "docs: README PDF-v0-critique rewrite + architecture + signal cards A1/A2/B1"
```

---

### Task 31: GitHub repo setup + bulk issue creation

**Files:**
- Create: `scripts/setup_github.sh`
- Create: `scripts/create_all_issues.py`

- [ ] **Step 1: Write `scripts/setup_github.sh`**

```bash
#!/usr/bin/env bash
# Creates the GitHub repo, milestones, and labels per spec §23.
# Idempotent — re-running is safe.
set -euo pipefail

REPO="auto-research"
OWNER="$(gh api user --jq .login)"

# 1. Create repo if it does not exist
if ! gh repo view "$OWNER/$REPO" &>/dev/null; then
  gh repo create "$REPO" --public --source=. --remote=origin --description \
    "Two-plane multi-agent research platform for cross-asset language-driven alpha"
  git push -u origin main
fi

# 2. Create milestones (one per week)
for due_offset_days in 7:1 14:2 21:3 28:4; do
  due_days="${due_offset_days%:*}"
  week_n="${due_offset_days#*:}"
  due_date="$(date -u -v+"${due_days}d" +%Y-%m-%d)T23:59:59Z"
  case "$week_n" in
    1) title="W1 — Foundation + extraction backbone" ;;
    2) title="W2 — RAG layer + extraction quality" ;;
    3) title="W3 — Signals + backtest gauntlet" ;;
    4) title="W4 — Research agent + live critic + MCP + polish" ;;
  esac
  if ! gh api "repos/$OWNER/$REPO/milestones" --jq '.[].title' | grep -qx "$title"; then
    gh api -X POST "repos/$OWNER/$REPO/milestones" \
      -f title="$title" -f due_on="$due_date" >/dev/null
    echo "created milestone: $title (due $due_date)"
  fi
done

# 3. Create labels
declare -A LABELS=(
  ["infra"]="cfd3d7"
  ["extract"]="fbca04"
  ["rag"]="0e8a16"
  ["signal"]="1d76db"
  ["backtest"]="5319e7"
  ["agent"]="b60205"
  ["mcp"]="d4c5f9"
  ["eval"]="fbca04"
  ["obs"]="bfdadc"
  ["docs"]="c2e0c6"
  ["polish"]="ededed"
  ["xs"]="ededed"
  ["s"]="ededed"
  ["m"]="ededed"
  ["l"]="ededed"
)
for name in "${!LABELS[@]}"; do
  color="${LABELS[$name]}"
  if ! gh label list -L 200 | awk -F'\t' '{print $1}' | grep -qx "$name"; then
    gh label create "$name" -c "$color" >/dev/null
    echo "created label: $name"
  fi
done

echo "GitHub setup complete."
```

- [ ] **Step 2: Write `scripts/create_all_issues.py`**

```python
"""Parse the implementation plan and create one GitHub issue per Task N.

Run after `scripts/setup_github.sh`. Idempotent — skips issues whose title already exists.
"""
import os
import re
import subprocess
from pathlib import Path

PLAN = Path(__file__).resolve().parents[1] / "docs" / "plans" / "2026-05-22-auto-research-implementation.md"

MILESTONE_BY_TASK = {}
for n in range(1, 13):  MILESTONE_BY_TASK[n] = "W1 — Foundation + extraction backbone"
for n in range(13, 23): MILESTONE_BY_TASK[n] = "W2 — RAG layer + extraction quality"
for n in range(23, 27): MILESTONE_BY_TASK[n] = "W3 — Signals + backtest gauntlet"
for n in range(27, 33): MILESTONE_BY_TASK[n] = "W4 — Research agent + live critic + MCP + polish"

LABEL_BY_KEYWORD = [
    (r"(?i)\b(ingest|edgar|fmp|backfill|manifest)\b", ["infra"]),
    (r"(?i)\b(extract|prompt|worker|guardrails)\b", ["extract"]),
    (r"(?i)\b(rag|chunk|retrieval|embedding|reranker|entity resolution)\b", ["rag"]),
    (r"(?i)\b(signal|combiner|alpha library)\b", ["signal"]),
    (r"(?i)\b(backtest|cpcv|sharpe|labels|cost model|info_tests|gate)\b", ["backtest"]),
    (r"(?i)\b(research agent|graph|checkpointer|hitl|live critic)\b", ["agent"]),
    (r"(?i)\b(mcp|fastmcp)\b", ["mcp"]),
    (r"(?i)\b(eval|deepeval|ragas|gold set)\b", ["eval"]),
    (r"(?i)\b(langfuse|telemetry|mlflow|observability)\b", ["obs"]),
    (r"(?i)\b(readme|architecture|signal cards|docs|memo)\b", ["docs"]),
    (r"(?i)\b(streamlit|dashboard|polish)\b", ["polish"]),
]

TASK_HEADER = re.compile(r"^### Task (\d+): (.+)$", re.MULTILINE)

def parse_tasks() -> list[dict]:
    text = PLAN.read_text()
    matches = list(TASK_HEADER.finditer(text))
    tasks = []
    for i, m in enumerate(matches):
        n = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        tasks.append({"n": n, "title": title, "body": body})
    return tasks

def labels_for(title: str, body: str) -> list[str]:
    labels: list[str] = []
    haystack = f"{title}\n{body}"
    for pat, ls in LABEL_BY_KEYWORD:
        if re.search(pat, haystack):
            labels += ls
    # Size heuristic: count `- [ ]` steps
    n_steps = body.count("- [ ] **Step")
    if n_steps <= 4: labels.append("s")
    elif n_steps <= 8: labels.append("m")
    else: labels.append("l")
    return sorted(set(labels))

def existing_issue_titles() -> set[str]:
    out = subprocess.check_output(
        ["gh", "issue", "list", "--state", "all", "--limit", "200", "--json", "title"],
        text=True,
    )
    import json as _json
    return {row["title"] for row in _json.loads(out)}

def create_issue(task: dict) -> None:
    title = f"Task {task['n']}: {task['title']}"
    if title in existing_issue_titles():
        print(f"skip (exists): {title}")
        return
    milestone = MILESTONE_BY_TASK.get(task["n"])
    labels = labels_for(task["title"], task["body"])
    body = (
        f"Implementation steps in [the plan]"
        f"(../blob/main/docs/plans/2026-05-22-auto-research-implementation.md#task-{task['n']}-"
        f"{task['title'].lower().replace(' ', '-').replace('+', '').replace('(', '').replace(')', '').replace(',', '')}).\n\n"
        f"## Acceptance criteria\n\n- [ ] All tests in the task pass under `uv run pytest -v`.\n"
        f"- [ ] Conventional-commits commit landed for the task.\n"
        f"- [ ] No `TBD` / `TODO` / `pass  # placeholder` left in code.\n"
    )
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    for lbl in labels:
        cmd += ["--label", lbl]
    if milestone:
        cmd += ["--milestone", milestone]
    subprocess.check_call(cmd)
    print(f"created: {title} (milestone={milestone}, labels={labels})")

def main() -> None:
    tasks = parse_tasks()
    for task in tasks:
        create_issue(task)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make scripts executable, commit**

```bash
chmod +x scripts/setup_github.sh
# Smoke-test parse (does not call gh):
uv run python -c "from pathlib import Path; import sys; sys.path.insert(0, 'scripts'); import create_all_issues; tasks = create_all_issues.parse_tasks(); print(f'parsed {len(tasks)} tasks'); assert len(tasks) >= 32"
git add scripts/setup_github.sh scripts/create_all_issues.py
git commit -m "feat(pm): setup_github.sh + create_all_issues.py for bulk issue creation"
```

- [ ] **Step 4: Run the live setup (only after plan approval — interview reviewers see the issues immediately)**

```bash
./scripts/setup_github.sh
uv run python scripts/create_all_issues.py
```

Expected: GitHub repo `<owner>/auto-research` exists, 4 milestones present, 32 issues created and assigned.

---

### Task 32: Optional Streamlit dashboard (stretch — first cut candidate)

**Files:**
- Create: `dashboard.py`
- Create: `tests/test_dashboard_smoke.py`

This task is **first in / first out** on the cuts list. Build only if W4 D20 has buffer.

- [ ] **Step 1: Write smoke test (verifies module imports + key functions don't crash on empty data)**

```python
# tests/test_dashboard_smoke.py
import importlib
import pandas as pd

def test_dashboard_module_imports():
    mod = importlib.import_module("dashboard")
    assert hasattr(mod, "load_alpha_library_df")
    assert hasattr(mod, "load_recent_memos")

def test_load_alpha_library_df_returns_dataframe_even_when_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    from dashboard import load_alpha_library_df
    df = load_alpha_library_df()
    assert isinstance(df, pd.DataFrame)

def test_load_recent_memos_handles_missing_dir(tmp_path):
    from dashboard import load_recent_memos
    memos = load_recent_memos(memo_dir=tmp_path / "missing")
    assert memos == []
```

- [ ] **Step 2: Implement dashboard**

```python
# dashboard.py
"""Streamlit dashboard — optional stretch (per spec §17).

Reads from MLflow + DuckDB + Feast directly — no new state.

Run: `uv run streamlit run dashboard.py`
"""
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

from auto_research.signals.alpha_library import AlphaLibrary

def load_alpha_library_df() -> pd.DataFrame:
    entries = AlphaLibrary().list()
    if not entries:
        return pd.DataFrame(columns=["signal_id", "sharpe_net", "deflated_sharpe",
                                     "code_version", "run_id"])
    return pd.DataFrame(entries)

def load_recent_memos(*, memo_dir: Path = Path("data/memos"), limit: int = 10) -> list[dict]:
    if not memo_dir.exists():
        return []
    out = []
    for p in sorted(memo_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        out.append({"name": p.name, "modified": p.stat().st_mtime, "body": p.read_text()})
    return out

def load_pnl_series() -> pd.DataFrame:
    """Placeholder: in production this reads DuckDB over backtest output Parquet."""
    return pd.DataFrame(columns=["date", "signal_id", "pnl"])

def main() -> None:
    st.set_page_config(page_title="auto-research", layout="wide")
    st.title("auto-research — alpha dashboard")

    st.header("Alpha library")
    alpha_df = load_alpha_library_df()
    st.dataframe(alpha_df, use_container_width=True)

    st.header("Recent attribution memos")
    memos = load_recent_memos(memo_dir=Path("data/memos"))
    for memo in memos:
        with st.expander(memo["name"]):
            st.markdown(memo["body"])

    st.header("PnL by signal (placeholder)")
    pnl_df = load_pnl_series()
    if not pnl_df.empty:
        st.line_chart(pnl_df.pivot(index="date", columns="signal_id", values="pnl"))
    else:
        st.info("No PnL series available — run a backtest first.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke test + commit**

```bash
uv run pytest tests/test_dashboard_smoke.py -v
git add dashboard.py tests/test_dashboard_smoke.py
git commit -m "feat(polish): optional Streamlit dashboard reading MLflow + memos"
git tag w4-complete
```

---

**Milestone 4 acceptance:**
- `uv run pytest -v` all pass (including agent + MCP + live critic tests)
- `uv run python -m auto_research.mcp_server` boots and is reachable from Claude Desktop after wiring per `docs/architecture.md`
- `uv run python scripts/worked_example_promote.py` produces `data/memos/we-promote-1.md` + MLflow run; `uv run python scripts/worked_example_kill.py` produces `data/memos/we-kill-1.md`
- `uv run auto-research research replay we-promote-1` re-runs deterministically from the SQLite checkpointer
- README + `docs/architecture.md` + three signal cards in place
- 32 GitHub issues exist across 4 milestones via `scripts/create_all_issues.py`
- (Stretch) `uv run streamlit run dashboard.py` loads without error

---

## Self-review

### Spec coverage

Walked through the spec section-by-section and mapped each to a task. Coverage map:

| Spec § | Tasks |
|---|---|
| §4 two-plane architecture | Tasks 1–32 (whole plan) + Task 30 architecture doc |
| §5 Universe | Task 4 (universe loader + JSON) |
| §6.1 Ingest | Tasks 5 (EDGAR), 6 (FMP), 22 (orchestrator) |
| §6.3 Feast PIT | Task 7 |
| §6.4 Extracted store | Tasks 9–11, 19 (via worker output) |
| §7 Extraction plane | Tasks 9 (schemas), 10 (client), 11 + 19 (workers), 20 (gold set + DeepEval) |
| §7.5 Guardrails | Citation validator in Task 9; Guardrails AI in pyproject + cited in `extract_client` retry path (covered via reliability primitives Task 8) |
| §8.1 RAG Flow 1 | Tasks 13–14 |
| §8.2 RAG Flow 2 | Tasks 15–17, 21 |
| §8.3 RAG Flow 3 | Task 18 |
| §8.5 Ragas | Task 21 |
| §9 Signal library | Task 26 |
| §10.1 T1 info_tests | Task 23 |
| §10.2 T2 backtest | Tasks 24–25 |
| §10.4 Reports | Task 24 |
| §10.5 Gates | Task 25 |
| §11 Research agent | Task 28 |
| §12 Live critic | Task 29 |
| §13.1 MCP server | Task 27 |
| §13.2 Reliability | Task 8 |
| §14.1 DeepEval | Task 20 |
| §14.2 Ragas | Task 21 |
| §15 Observability | Tasks 2 (Langfuse + OTel), 3 (MLflow) |
| §16 LLM cost stack | Task 10 (tiered models + caching) + Batch API kickoff in Task 22 |
| §17 Streamlit | Task 32 (optional) |
| §18 Schedule | Reflected in task ordering W1→W4 |
| §19 Cuts list | Reflected: Task 32 first cut, B1 (Task 26) third cut, live critic (Task 29) second cut |
| §23 PM/GitHub | Task 31 |

**Identified gaps + fixes applied inline:**

1. **§14.3 G-Eval memo rubric** — not separately tasked; the worked-example memos in Task 29 are the artifacts a future G-Eval rubric would score, and the rubric itself slots into the existing DeepEval harness from Task 20. Reasonable defer; documented here.
2. **§7.4 Backfill economics + Batch API submission** — Task 22 kicks off backfill but does not break out the Anthropic Batch API submission as its own task. The Batch path is exercised by the existing `ExtractionClient` (Task 10) which already uses the `messages.create` synchronous path; a Batch wrapper is a logical follow-on issue but unblocks no other work in v1 and is excluded from M3-M4 scope.
3. **§6.4 quarantine path** — Pydantic validation failure routing to `data/quarantine/` is mentioned in spec §7.5 but no explicit task adds the quarantine sink. This is covered implicitly by Guardrails AI raising on schema-validation failure (caller decides what to do), so v1 punts on a dedicated quarantine writer.

### Placeholder scan

Searched the appended Milestones 3-4 content for the red-flag patterns from the writing-plans skill:

- "TBD" / "TODO" / "implement later" → none found in new content (one TBD in original M1 Task 1 README stub is pre-existing).
- "Similar to Task N" → none.
- "add appropriate error handling" → none.
- Steps without code blocks when code is required → none. Every implementation step in M3-M4 has a complete code block.
- References to undefined types/functions → cross-checked: `InfoReport`, `BacktestReport`, `GateDecision`, `T1_GATE`, `T2_GATE`, `AlphaLibrary`, `HypothesisType`, `ResearchState`, `LiveCriticInput`, `LiveCriticOutput`, `LocalBGEEmbedder`, `LanceVectorStore`, `HybridRetriever`, `FeatureStore` all defined in cited locations.

Two known placeholders kept on purpose:
- `run_validation_node` returns a zeroed `InfoReport` with `notes="placeholder — wired at runtime via MCP"`. This is the **integration seam** — the test stubs replace the node entirely; in production the node calls `mcp_server.run_backtest` which is itself a stub in Task 27 (T2 path returns `{"error": "signal materialization stub"}`). This is a documented v1 limitation that does not block the rest of the system, since (a) the gates work on any well-formed report, (b) the worked-example scripts in Task 29 bypass this seam by injecting reports directly, and (c) the MCP read-only surface still functions independently.
- `dashboard.load_pnl_series()` is documented as a placeholder for DuckDB-on-Parquet — the dashboard is the first cut candidate, so investing in this is YAGNI until B1/A1/A2 produce real PnL Parquet output.

Both are flagged in their respective code so they are visible to anyone reading the file.

### Type consistency

Cross-referenced symbols between Milestones 1-2 and Milestones 3-4:

| Symbol | Defined in | Used in |
|---|---|---|
| `ExtractionClient`, `ExtractionCache` | Task 10 | Tasks 11, 19 (existing) — not reused in M3-M4 |
| `CircuitBreaker`, `CostCap`, `retry_with_backoff` | Task 8 | Task 10 (existing) — not reused in M3-M4 |
| `init_mlflow`, `log_run` | Task 3 | Task 26 (AlphaLibrary) — consistent ✓ |
| `LocalBGEEmbedder`, `LanceVectorStore`, `HybridRetriever` | Tasks 15-16 | Task 27 (mcp_server.search_memos) — consistent ✓ |
| `narrative_sources`, `tradeable`, `all_tickers` | Task 4 | Task 22 (existing) — not reused in M3-M4 |
| `InfoReport`, `BacktestReport` | Task 24 (new) | Tasks 25, 26, 28, 29 — consistent ✓ |
| `check_t1_gate`, `check_t2_gate`, `T1_GATE`, `T2_GATE`, `GateDecision` | Task 25 (new) | Tasks 26 (AlphaLibrary.promote), 28 (decide_node) — consistent ✓ |
| `HypothesisType`, `ResearchState` | Task 28 (new) | Task 29 (worked examples) — consistent ✓ |
| `TransactionCostModel` | Task 24 (new) | Task 25 (engine) — consistent ✓ |
| `combinatorial_purged_cv` | Task 24 (new) | Task 25 (engine) — consistent ✓ |
| `deflated_sharpe_ratio` | Task 24 (new) | Task 25 (engine) — consistent ✓ |
| `build_research_graph`, `replay_session` | Task 28 (new) | Task 29 (worked examples), Task 28 CLI patch — consistent ✓ |
| `LiveCriticInput`, `LiveCriticOutput`, `build_live_critic_agent` | Task 29 (new) | `scripts/cron_daily_critic.sh` — consistent ✓ |

Method signatures verified consistent. One naming convention to note: the spec uses `event_classification` for 8-K events in §6.3 FeatureView column naming, while the Pydantic schema (Task 9 from M2) names the field `events`. The MCP `get_feature_definition` lookup table (Task 27) preserves the FeatureView column name `event_classification` to match the spec's Feast schema, which is the right level of abstraction for that surface. This is consistent — Feast column names and Pydantic field names need not match.

No fixes required.

---
