# Observability for AI engineering — a tutorial

> Self-study material. Saved verbatim from a session walking through the
> deeper concepts behind `src/auto_research/telemetry.py`. Target reader:
> an AI engineer who can already explain a stack trace and a span and
> wants the LLM-specific scaffolding around it. Interview-prep depth.

---

## 1. Why observability is harder for AI/LLM systems than for normal software

Normal observability answers: *did this request succeed, how slow was it, where did it fail?* The three pillars (logs, metrics, traces) plus error tracking get you 90% of the way there.

LLM systems add several axes that traditional observability stacks don't address natively:

| New axis | What you actually want to see |
|---|---|
| **Cost** | $ per request, per node, per session, per user. Token counts × tier prices. |
| **Quality** | Did the response answer the question correctly? Did it hallucinate? Pass an eval? |
| **Determinism** | Same prompt + same model version + same input → same output? Or non-deterministic regression? |
| **Cache attribution** | Was this call served from prompt cache? At what % discount? |
| **Multi-step traces** | A single user request triggers 4 LLM calls + 2 retrievals + 1 tool call. Correlate. |
| **Prompt provenance** | Which prompt version produced this output? What changed between v3 and v4? |
| **Model staleness** | Production calls still routed to claude-sonnet-4-5 after the 4-6 upgrade should fail loud. |
| **PII / content sensitivity** | Prompts contain customer data. Where does it go? Who can read traces? |
| **Drift** | Is the response distribution shifting silently (e.g., model gets more verbose, costs creep up)? |

Traditional APM (Datadog, New Relic) treats an LLM call as just another HTTP call to `api.anthropic.com`. You see latency and status, not the prompt, response, or token attribution. That's why a dedicated LLM observability layer exists.

---

## 2. The three pillars, stretched for LLM

### Logs

Still the same: timestamped text. New flavor: *prompt content* and *raw response* are large multi-line strings, very expensive to dump per call. Most LLM platforms attach prompts/responses to **spans** instead of separate logs — they're structured (model, role, tokens) and queryable.

Rule of thumb: if you'd log a prompt with `logger.info(prompt)`, you should *attach it to a span* instead. Logs scattered across processes are hard to correlate; spans carry trace context for free.

### Metrics

Counters and histograms over time. The LLM-specific metrics:

- `llm.tokens.input` and `llm.tokens.output` — counters, by model + by route
- `llm.latency` — histogram, p50/p95/p99 by model
- `llm.cost_usd` — counter, derived from tokens × tier prices
- `llm.cache.hit_rate` — ratio gauge
- `llm.errors{type}` — counter labeled by `rate_limit`, `overload`, `timeout`, `content_policy`, etc.
- `llm.retries{reason}` — counter

The cardinality trap: if you put `prompt_id` or `user_id` as a metric label, your TSDB explodes. Put high-cardinality identifiers in **trace attributes**; keep metric labels low-cardinality (model name, status, tier).

### Traces (the LLM workhorse)

A single user action produces a tree of spans:

```
[span] user_request
  ├─[span] research.propose_hypothesis    (LLM call — Sonnet)
  │   ├─[span] memo_retrieval.search       (RAG call — embeddings + LanceDB)
  │   └─[span] anthropic.messages.create
  ├─[span] research.run_validation
  │   ├─[span] backtest.engine.run
  │   └─[span] anthropic.messages.create   (Critic — Sonnet)
  └─[span] research.write_memo
      └─[span] anthropic.messages.create   (Writer — Haiku)
```

Each LLM-call span carries the prompt, response, model, token counts, cost, latency, cache stats. You can answer:

- "Why did this user-request cost $0.34? Which sub-call dominated?" → drill into the trace.
- "Was the response in span X grounded in the retrieval in span Y?" → click both, compare.
- "Why did this trace fail?" → red span shows the exception, parent chain shows context.

This is the central abstraction. Master tracing first; metrics and logs become afterthoughts.

---

## 3. OpenTelemetry primer (the substrate)

OTel is the W3C-blessed standard for traces + metrics + logs. Vendor-neutral. The pieces:

| Concept | What it is |
|---|---|
| **Tracer** | Factory for spans. `tracer = trace.get_tracer(__name__)` |
| **Span** | A unit of work with start/end timestamps + attributes (k=v dict) + events + status. |
| **Trace** | A tree of spans sharing a trace ID. |
| **Context propagation** | How a child span knows its parent. Via thread-local in-process, via headers across processes. |
| **TracerProvider** | Process-global config: which sampler, which exporter, which resource. |
| **Resource** | Attributes that apply to every span from this process (`service.name="auto-research"`, `service.version="0.1.0"`). |
| **Sampler** | Per-trace decision: record or drop. (For LLM: usually 100% sampling since traces are sparse-but-expensive.) |
| **SpanProcessor** | Pipeline stage: receives spans, decides what to do (e.g., batch them). |
| **Exporter** | Final destination: OTLP/HTTP, OTLP/gRPC, Jaeger, console, etc. |
| **OTLP** | The wire protocol. Two flavors: gRPC (port 4317) and HTTP/protobuf (port 4318). Langfuse speaks HTTP. |

### Two ways to instrument

**Manual:**

```python
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("propose_hypothesis") as span:
    span.set_attribute("hypothesis.type", "conditional")
    result = run_hypothesis(...)
    span.set_attribute("hypothesis.result", result.status)
```

**Auto-instrumentation:** a library monkey-patches client SDKs (Anthropic, OpenAI, LangChain, requests) so every call produces a span automatically — that's what OpenLLMetry / OpenInference do for you. You added zero manual `start_as_current_span` calls in your code; the spans appear when you call `client.messages.create()`.

**Best practice: combine.** Auto-instrumentation for the LLM/RAG/HTTP layers. Manual spans for *domain* nodes (`research.propose_hypothesis`, `backtest.run_t2`). Domain spans become the parents that auto-spans nest under, giving you the tree.

### Batch vs simple exporters

Spans don't go straight over the wire. The `BatchSpanProcessor` buffers them and flushes:

- Every N spans (default 512)
- Or every M ms (default 5000)
- On shutdown

That's why `force_flush()` exists. In a long-running daemon, batching is fine. In a CLI/test that exits immediately, you'd lose the last batch — so the integration test in this codebase calls `provider.force_flush(timeout_millis=5000)` before querying Langfuse.

For unit tests where you want synchronous span access, use `SimpleSpanProcessor` (no batching). Trade-off: 1 HTTP call per span. Fine for tests, terrible for prod.

### Context propagation across services

Suppose your research agent makes an HTTP call to your MCP server. Without explicit propagation, the MCP server's spans land in a separate trace. With propagation:

```python
# in research agent
import requests
from opentelemetry.propagate import inject

headers = {}
inject(headers)  # sets traceparent, tracestate headers
requests.get(mcp_url, headers=headers)
```

```python
# in MCP server
from opentelemetry.propagate import extract
ctx = extract(request.headers)
with tracer.start_as_current_span("mcp.query_features", context=ctx) as span:
    ...
```

The auto-instrumentation for `requests` does this for you. For raw `httpx` or async libs, sometimes you need to do it manually. **Interview signal:** can you explain `traceparent: 00-<trace-id>-<span-id>-01` (the W3C trace context format)? It's just a string in a header.

---

## 4. OpenInference + Semantic Conventions for LLMs

OTel knows what an HTTP call looks like (`http.method`, `http.status_code`). It doesn't natively know what an LLM call looks like (`llm.prompt`, `llm.model`).

**OpenInference** (Arize) and **OTel GenAI Semantic Conventions** (the official W3C effort, still partly experimental) define a vocabulary of span attribute names for LLM operations:

| Attribute | Meaning |
|---|---|
| `llm.model_name` | "claude-sonnet-4-6" |
| `llm.system` | "anthropic" |
| `llm.invocation_parameters` | JSON: `{temperature, max_tokens, ...}` |
| `llm.prompt_template.template` | The raw template (before variable substitution) |
| `llm.prompt_template.variables` | The variables that filled it in |
| `llm.token_count.prompt` | Input tokens |
| `llm.token_count.completion` | Output tokens |
| `llm.token_count.total` | Sum |
| `input.value` | The full rendered prompt (multi-message JSON) |
| `output.value` | The full response |
| `openinference.span.kind` | "LLM" / "EMBEDDING" / "RETRIEVER" / "RERANKER" / "AGENT" / "CHAIN" / "TOOL" |

When OpenLLMetry instruments Anthropic, it produces spans following the OpenInference conventions. Langfuse understands those conventions and renders them as: model badge, prompt/response panels, token counts at the top, cost calculation, latency.

**Why two standards?** OTel GenAI is the official long-term track (slow, careful, draft). OpenInference is the de-facto current standard (Arize-led, faster-moving, what tools actually consume today). They're converging.

**Interview answer template:** "We use OpenInference span conventions because that's what Langfuse + Arize + most LLM tooling parse today. The OTel GenAI semantic conventions are the long-term direction once they stabilize."

---

## 5. The LLM observability stack — from your code to the dashboard

Concretely, what happens once `init_telemetry()` has run:

```
┌─────────────────────────────────────────────────────────────┐
│ your code: anthropic.Anthropic().messages.create(...)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ OpenLLMetry monkey-patch     │   = wraps the SDK method
        │ (traceloop-sdk)              │   = creates OTel spans
        │                              │     with OpenInference attrs
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ OpenTelemetry SDK            │   = global TracerProvider
        │ - Tracer                     │   = stores resource attrs
        │ - BatchSpanProcessor         │   = batches and flushes
        └──────────────┬──────────────┘
                       │  spans (in-memory)
        ┌──────────────▼──────────────┐
        │ OTLP HTTP Exporter           │   = serializes to protobuf
        │ (POST + Auth: Basic ...)     │   = HTTP POST to Langfuse
        └──────────────┬──────────────┘
                       │  protobuf over HTTP
                       │  http://localhost:3000/api/public/otel
        ┌──────────────▼──────────────┐
        │ Langfuse ingestion endpoint  │   = parses OTLP
        │                              │   = normalizes attrs
        │                              │   = stores in Postgres (+ S3
        │                              │     for big content blobs)
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ Langfuse UI / API            │   = trace tree, prompt/response
        │ http://localhost:3000        │     viewer, cost rollup,
        │                              │     eval scores, prompt registry
        └─────────────────────────────┘
```

Every piece is swappable:

- Replace `traceloop-sdk` with `openinference-instrumentation-anthropic` for direct OpenInference + OTel.
- Replace OTLP HTTP exporter with OTLP/gRPC (different port, slightly faster).
- Replace Langfuse with Arize Phoenix, LangSmith, Helicone, Honeycomb, or DIY Postgres — they all speak OTLP.

This is what "vendor-neutral instrumentation" buys you: your code calls don't know where the spans go.

---

## 6. The pillars of LLM observability (the framework to memorize)

Most senior interviewers want this framework. Build the mental model around five concerns:

### Pillar 1: Tracing (single-call debuggability)

"This one call: what happened?"

- Full prompt + full response visible
- Token counts populated
- Cost calculated
- Latency broken down (network vs API vs render)
- Errors with span status + exception event
- Parent chain visible (caller context)

Toolwise: Langfuse, LangSmith, Phoenix, Helicone all do this.

### Pillar 2: Evaluation (quality measurement)

"Is the response *good*?"

LLM responses don't have a `200 OK`. You need a *judgment* — either deterministic (regex match, JSON-schema validation, citation-grounding) or LLM-judge (G-Eval, Ragas faithfulness).

Modern observability platforms attach eval scores to traces:

- Run an LLM-judge over recent traces
- Score 0-5 on `relevance`, `faithfulness`, `citation_quality`
- Filter the UI: "show me traces where faithfulness < 3"

This blurs the line between observability and testing. Eval-as-observability is the 2025 frontier.

### Pillar 3: Cost (financial observability)

"What did this user/session/feature *cost*?"

Token counts per call + model tier prices = dollar amount. Aggregated by:

- Per request (debugging expensive single requests)
- Per session (researcher running 50 hypotheses in a day)
- Per feature (extraction vs research-agent)
- Per model (Haiku vs Sonnet vs Opus)

Cache hit rate is a big lever. Cached prompt input tokens are ~10% of the price; tracking `cache_read` vs `cache_creation` vs `cache_miss` shows where caching saves money.

### Pillar 4: Prompt registry + versioning

"Which prompt produced this output? What changed between v3 and v4?"

A registry (Langfuse, Promptlayer, custom) stores prompt templates by version. Spans carry `llm.prompt_template.template_id="v4"`. When you bump a prompt, you can:

- A/B comparison: run v3 and v4 against the same eval set
- Regression: filter recent traces by `template_id=v3` and `template_id=v4` and compare eval-score distribution
- Audit: which exact prompt was in production at 2026-04-15 14:30?

The `bump-prompt-version` skill in this repo is the mechanical defense for this. The registry is the observability surface.

### Pillar 5: Drift / regression detection

"Is the system getting worse over time?"

This is the hardest pillar. Distributions you watch:

- Response length distribution (mean, p95 token counts over time)
- Cost per request (creeping up = something is calling more often or with longer prompts)
- Eval-score distribution per prompt version
- Cache hit rate
- Tool-call failure rate
- Latency p99
- Specific failure patterns ("the word 'cannot' appearing in 3% of outputs last month → 12% this week")

Tooling for this is immature. Phoenix and Langfuse have basic dashboards; serious systems push data to Snowflake or BigQuery and build their own.

**Interview answer template:** "Observability for LLM systems means five pillars: tracing for single-call debug, evals for quality, cost attribution, prompt versioning + registry, and drift detection. Most tools handle the first three; evals and drift are where the frontier is."

---

## 7. Tool landscape with tradeoffs

This project picked Langfuse. Defend that and know the alternatives.

| Tool | OSS? | Self-host? | Strengths | Weaknesses |
|---|---|---|---|---|
| **Langfuse** | yes (MIT) | yes (Docker) | Self-hostable, OTLP standard, prompt registry, eval scoring, good UI, prices content-blob storage well | v3 stack heavyweight (ClickHouse + Redis + worker); v2 simpler but legacy |
| **LangSmith** | no | no (SaaS) | Polished UI, native LangChain hooks, fast | Vendor lock-in, paid, your prompts on their cloud |
| **Arize Phoenix** | yes (Elastic 2.0) | yes | OpenInference home base, strong eval/drift tooling, embedded use | Smaller ecosystem; UI less mature than Langfuse |
| **Helicone** | yes | yes | Proxy-based (no SDK change), simple, cheap | Proxy adds latency; less rich than span-based tools |
| **Honeycomb** | no | no | Powerful query language (BubbleUp, heatmaps), excellent for distributed traces | $$ per event; not LLM-specific; you build the conventions |
| **Datadog LLM Observability** | no | no | Enterprise integration with rest of DD APM | Expensive; LLM module is recent |
| **W&B Traces** | no | partial | Strong if you already use W&B for training | LLM-trace UX less polished than Langfuse |
| **Braintrust** | no | no | Eval-first product; great for prompt iteration | Less trace-focused; vendor lock-in |
| **DIY (OTel → Tempo/Grafana)** | yes | yes | Total control, no per-event pricing | You build prompt registry, eval, drift dashboards |

**Decision framework for interviews:**

- "Public-repo interview project, self-hosting matters, want OSS, want prompt registry built-in" → Langfuse (this project's choice).
- "Heavy LangChain user, willing to pay, want zero ops" → LangSmith.
- "Eval discipline > tracing polish, embedded systems" → Phoenix.
- "Already on Honeycomb for backend traces, want unified view" → Honeycomb + custom semantic conventions.
- "Want minimum-friction proxy, OpenAI-only" → Helicone.

The interview value isn't picking the "right" tool — it's articulating the tradeoff cleanly.

---

## 8. Likely interview questions with answer templates

### Q: "Walk me through what happens when your agent makes an LLM call."

> Three layers. (1) The Anthropic SDK gets monkey-patched by OpenLLMetry at process start — `Traceloop.init()` does that. (2) Each `messages.create()` call now produces an OpenTelemetry span with OpenInference attribute conventions: `llm.model_name`, `llm.token_count.prompt`, `input.value`, `output.value`, etc. (3) Spans go through a `BatchSpanProcessor` that buffers and flushes as protobuf-over-HTTP to Langfuse's OTLP endpoint at `/api/public/otel`, authenticated with Basic auth derived from the Langfuse public + secret keys. Langfuse stores them in Postgres, indexes for query, renders the trace tree in the UI. Same call inside a parent span (e.g., `research.propose_hypothesis`) nests automatically via in-process context propagation.

### Q: "How do you correlate cost back to a business action?"

> Cost lives at the leaf span (the LLM call). The business action is a manual parent span — `with tracer.start_as_current_span("research.propose_hypothesis"): ...`. OTel propagates context, so all child spans (LLM, retrieval, tool calls) share the same trace ID and parent chain. Aggregate by trace ID, sum `llm.token_count.total × price-per-token-by-model`. Surface as a per-trace cost in the UI and as a metric `llm.cost_usd{action="propose_hypothesis"}`.

### Q: "How do you detect prompt regression after a prompt bump?"

> Three steps. (1) Prompt version is a span attribute (`llm.prompt_template.template_id="v4"`). (2) Eval scores are attached to traces — either inline (a span event with `eval.faithfulness=0.83`) or as a post-hoc scoring job. (3) Compare the score distribution between `template_id="v3"` and `template_id="v4"` over a fixed test set. Wilcoxon signed-rank if you're being rigorous. Block the prompt-bump PR if median score regresses beyond a threshold. The version bump itself is mechanized with a repo skill so the version is *always* updated when the template changes — without that, the cache-hit / regression-detection contract is dead.

### Q: "Why Langfuse and not LangSmith?"

> Three reasons. (1) Self-hostable means prompts and customer-data-flavored content stay on our box, not LangChain's cloud — important for any data-sensitive deployment. (2) OTLP-based ingest means our instrumentation is vendor-neutral; we could swap Langfuse for Phoenix or DIY without changing application code. LangSmith's SDK locks you in. (3) Built-in prompt registry + eval scoring + cost tracking, all OSS. Tradeoff: Langfuse's UI is a step behind LangSmith's and the v3 stack (ClickHouse + Redis + worker) is heavier than v2. We're on v2 for now.

### Q: "What's the difference between OpenInference and OpenTelemetry?"

> OpenTelemetry is the foundation: span format, trace propagation, exporters, sampling. Domain-agnostic. OpenInference is a *vocabulary* — a set of semantic conventions defining what attributes an LLM-call span carries (`llm.model_name`, `input.value`, `openinference.span.kind="LLM"`). OpenLLMetry (traceloop-sdk) is an instrumentation *library* that emits OpenInference-compliant spans for LLM SDKs. So: OTel is the protocol, OpenInference is the schema, OpenLLMetry is the auto-instrumentor. They stack.

### Q: "How do you handle PII in traces?"

> Three controls. (1) Disable prompt/response content capture by default at the instrumentation level — Traceloop has a `TRACELOOP_TRACE_CONTENT=false` flag, OpenInference has similar. Only enable in environments where the data is sanitized. (2) For prod where some capture is necessary, add a span processor that scrubs known patterns (regex for emails, credit-card numbers) before export. (3) Restrict who can query traces — Langfuse projects + role-based access. The architectural point: prompts ARE customer data; treat the trace store with the same scrutiny as your DB.

### Q: "Sampling strategies for LLM traces?"

> LLM traces are sparse (vs millions of web requests) and expensive (each is rich). Default to 100% sampling unless cost is prohibitive. When you do sample:
> - Tail-sampling > head-sampling: keep all traces with errors, with high latency, with high cost. Drop the boring 99%.
> - Probabilistic sampling for high-volume systems with low-variance traffic (e.g., a chatbot with millions of identical greetings).
> - Always-on for low-volume high-value flows (the research agent).

### Q: "Why is observability harder for LLM systems than for normal services?"

> Five new axes: cost, quality, determinism, prompt provenance, drift. None of which traditional APM addresses. Plus the *content* is the payload and the content is huge and possibly sensitive — different from logging an HTTP request body. That's why a dedicated LLM-obs layer exists on top of OTel rather than just slapping Datadog on it.

---

## 9. Anti-patterns (what *not* to say in an interview)

| Anti-pattern | Why it's wrong |
|---|---|
| "We just log the prompt and response with `logger.info`" | Loses context propagation, hard to query, expensive at scale, no cost attribution. |
| "We instrument manually inside every LLM call site" | Brittle. Easy to forget. Use auto-instrumentation for SDKs; manual only for domain spans. |
| "Sampling at 10% to save money" | LLM traces are sparse and expensive each — sampling saves the wrong axis. Tail-sample instead. |
| "We mock the LLM in tests so traces work" | Mocks defeat the test. Use real LLM in eval/integration suites; mock in unit suites only. |
| "Eval scores live in a separate spreadsheet" | They should be span attributes / scores attached to traces, queryable alongside cost + latency. |
| "Prompt versions are tracked in git commits" | True but not queryable from observability. Need the version as a span attribute too. |
| "Prompts/responses go to a third-party SaaS, no review" | PII risk. Self-hosted obs or strict scrubbing required for sensitive data. |
| "We rely on the LLM provider's dashboard (Anthropic console)" | Provider dashboard shows their view, not your call-tree view. You need your own. |

---

## 10. The frontier: where this is going (good for the "what excites you" question)

- **Eval-as-observability:** traces continuously scored by an LLM-judge, regressions auto-detected. Phoenix and Langfuse both moving here.
- **Distributed eval pipelines:** run G-Eval over a streaming trace feed (Kafka + LLM-judge) for live quality dashboards.
- **Multi-model attribution:** "Which model is cheapest for this category of request?" — automatic routing based on observed eval × cost.
- **Replay + counterfactual:** "What would v5 of this prompt have produced on the last 1000 production calls?" — replay traces against an alternate prompt and diff.
- **Embedding-based drift:** detect distribution shift in prompts or responses via embedding clustering over time.
- **OTel GenAI semantic conventions stabilization:** OpenInference will likely fold into OTel official ~2026.

If asked "what's exciting," pick one of these and have a concrete take. ("Eval-as-observability is the unblocker — once your faithfulness score is a queryable span attribute and CI gates on it, prompt iteration becomes safe instead of a guessing game.")

---

## Suggested learn-order (for genuine depth, not just interview prep)

1. **Read OpenTelemetry Python docs** end-to-end. ~3 hours. The trace/span/exporter mental model is the foundation; without it everything else is hand-waving.
2. **Read this repo's `src/auto_research/telemetry.py` plus the installed `traceloop/sdk/__init__.py`.** Connect what the docs taught you to a real ~150-line implementation. ~1 hour.
3. **Run the integration test, open Langfuse, click into the trace.** See the Anthropic call with the prompt panel, response panel, token counts. ~30 min.
4. **Read the OpenInference semantic conventions spec.** Short, ~30 min.
5. **Try Arize Phoenix locally** (`pip install arize-phoenix`, `phoenix serve`). Point `init_telemetry` at it instead of Langfuse. Watch the same traces render differently. Confirms vendor-neutrality is real, not a slogan. ~1 hour.
6. **Pick one of the 5 pillars** (probably drift or eval) and dig: read 2 vendor blog posts + 1 academic paper. ~2 hours.

Total: ~7-8 hours to be genuinely fluent. After that you're not memorizing answers — you're reasoning from a stable mental model and any question lands in known territory.
