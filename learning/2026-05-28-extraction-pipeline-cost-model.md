# Extraction pipeline cost model — Anthropic API economics, local-inference trade-offs

> Captured: 2026-05-28, from PR #77 review + the follow-up discussions in
> issues #78 (tool_use / Option A / cleanup) and #79 (complete TenKFinancials).
>
> This is teaching material — the actual cost build-up behind the numbers
> quoted in those issues, the assumptions baked in, and the decision
> framework for "when should we move to local inference?". Anthropic pricing
> evolves; treat the absolute dollars as a snapshot and the *ratios* +
> *reasoning* as the durable content.

---

## 1. Why a documented cost model matters

### 1.1 The problem

Extraction pipelines that read SEC filings and earnings transcripts make
hundreds to thousands of LLM calls per backfill cycle. Per-call cost is
small (sub-cent), but two things make it easy to misjudge total cost:

1. **Multiplicative fan-out.** One filing → multiple workers (narrative,
   Item 8 financials, contextual chunking per child chunk) → each may
   issue multiple LLM calls. A "small per-call cost" times "many calls"
   creates surprising totals.
2. **Routing-table drift.** If the spec says "use Haiku for templated
   tasks, Sonnet for cross-doc reasoning," but the implementation
   silently routes everything to Sonnet, real cost is 5× the design
   intent. The bug is invisible without a cost model to compare against.

The wrong question is "how much does Claude cost?" — that's a per-token
number with no operational meaning. The right question is "what do *we*
spend per backfill cycle, decomposed by worker, by model tier, and by
call shape?" — that's what this document answers.

### 1.2 What this model is for

Three audiences:

- **Reviewers** evaluating a PR that touches the extraction layer
  (Option A, tool_use migration, etc.) — does the cost claim hold up?
- **Future maintainers** considering local inference, batch API
  adoption, or a routing-table change — what's the headroom?
- **Anyone debugging "why is our bill 3× last month?"** — which knob
  to look at first.

It's NOT for product/business cost forecasting at scale. The numbers
assume the current ~500-ticker universe; scaling 10× changes the
infrastructure decision (see §9).

---

## 2. Anthropic pricing as of 2026-05

### 2.1 Per-token prices

| Tier | Input | Cache read | Cache write | Output |
|---|---|---|---|---|
| Haiku 4.5 | $1.00 / MTok | $0.10 / MTok | $1.25 / MTok | $5.00 / MTok |
| Sonnet 4.6 | $3.00 / MTok | $0.30 / MTok | $3.75 / MTok | $15.00 / MTok |
| Opus 4.7 | $15.00 / MTok | $1.50 / MTok | $18.75 / MTok | $75.00 / MTok |

Two relationships hold across tiers:

- **Output is 5× input** at every tier. So output-heavy calls (extended
  thinking, verbose JSON) are disproportionately expensive.
- **Cache read is 10% of input price**; cache write is **125%** of
  input price. The cache pays back on the second call within the 5-minute
  TTL — first call is *more* expensive than uncached because of the
  write surcharge, but every subsequent call within the window is 10×
  cheaper than uncached.

### 2.2 The 5-minute TTL is the load-bearing constraint

A cached prefix only stays warm for 5 minutes. This matters because:

- **Backfill orchestration** must batch per-worker calls within 5-minute
  windows or pay the cache-write surcharge repeatedly.
- **Cross-worker amortization doesn't work** — each worker has its own
  system prompt, so the cached prefix is worker-specific. Item 8 calls
  don't warm the narrative cache.
- **Production rollout cadence** matters: bumping a `*_PROMPT_VERSION`
  invalidates the cache instantly. The first call after deploy pays
  the write surcharge again.

### 2.3 The Batch API discount

Anthropic Batch (`messages.batches.*`) is **50% off everything** —
input, output, cache read, cache write. The trade-off: requests run
within 24 hours, not synchronously. For backfill (which is throughput-
sensitive, not latency-sensitive), this is the single biggest cost
lever that doesn't require any code change beyond flipping a config
flag in the orchestrator.

Every dollar figure in this document **assumes synchronous API**. Halve
everything if the call goes through Batch.

---

## 3. Per-call cost — the unit economics

### 3.1 Shape parameters that drive cost

Three numbers determine a call's cost:

1. **Cached input** (system prompt + few-shot examples + prior cached
   user blocks). Subject to the 10% cache-read discount on warm hits.
2. **Fresh input** (per-document user content — the raw filing text or
   RAG-retrieved parents). Always full price.
3. **Output** (the model's response — JSON + extended thinking blocks
   if enabled).

Extended thinking budget is billed as output tokens, so a
`budget_tokens=2048` thinking-enabled call adds ~$0.031 to a Sonnet
call regardless of how much actual JSON the model emits.

### 3.2 Typical call shapes in this codebase

Assuming warm cache (the typical state during a batch backfill — calls
land within 5 minutes of each other):

| Call shape | Cached prefix | Fresh input | Output | Per-call |
|---|---|---|---|---|
| Item 8 financials (Haiku) | 1500 tok × $0.10 = $0.00015 | 1000 tok × $1.00 = $0.0010 | 1000 tok × $5.00 = $0.0050 | **~$0.006** |
| Item 8 classifier (#79, Haiku) | 800 tok × $0.10 = $0.00008 | 500 tok × $1.00 = $0.0005 | 200 tok × $5.00 = $0.0010 | **~$0.002** |
| Narrative single-field (Haiku, RAG) | 1500 tok × $0.10 = $0.00015 | 10000 tok × $1.00 = $0.0100 | 1000 tok × $5.00 = $0.0050 | **~$0.015** |
| Narrative single-field (Sonnet, RAG) | 1500 tok × $0.30 = $0.00045 | 10000 tok × $3.00 = $0.0300 | 1000 tok × $15.00 = $0.0150 | **~$0.045** |
| Narrative fused (Sonnet, current bug) | 1500 tok × $0.30 = $0.00045 | 10000 tok × $3.00 = $0.0300 | 3000 tok × $15.00 = $0.0450 | **~$0.075** |
| Sonnet + extended thinking | same | same | +2048 thinking tok × $15.00 = $0.031 | **+~$0.031** |
| Contextual chunking (Haiku, one per child chunk) | 500 tok × $0.10 = $0.00005 | 500 tok × $1.00 = $0.0005 | 100 tok × $5.00 = $0.0005 | **~$0.001** |
| 8-K extraction (Haiku) | 1500 tok × $0.10 = $0.00015 | 2000 tok × $1.00 = $0.0020 | 500 tok × $5.00 = $0.0025 | **~$0.005** |

### 3.3 First-call surcharge (often ignored)

The first call in a 5-minute window pays the cache-write surcharge:

```
1500 tok × $1.25/MTok = $0.0019 surcharge
```

Across a typical batch of 5+ calls in one window, this amortizes to
~$0.0004 per call. Across an isolated single call (e.g., a one-off
test), it's the dominant cost component.

### 3.4 The cache-read counter is the canary

`response.usage.cache_read_input_tokens > 0` is the proof that the
cache is firing. A regression that strips `cache_control: ephemeral`
from the system block zeros this counter and triples per-call cost
silently. PR #77 added the corresponding OTel attribute
(`llm.cache_read_input_tokens`) precisely so this regression
*can't* happen invisibly.

---

## 4. Per-document cost — building up from per-call

### 4.1 Long 10-K (RAG path, ≥100K tokens)

A 10-K above the `SINGLE_SHOT_TOKEN_CUTOFF` runs the RAG path:
per-field retrieved-parent extraction for the narrative + per-table
extraction for Item 8.

**Today (post-PR #77 correctness, pre-Option A):**

| Cost line | Calls | Per-call | Subtotal |
|---|---|---|---|
| Narrative fused (Sonnet) — repeated 5× per Option A bug | 5 | $0.075 | $0.375 |
| Item 8 (one prompt per table × 5 tables) | 5 | $0.006 | $0.030 |
| **Total** | | | **~$0.41** |

The narrative line dominates. The 5× fan-out is the cost of asking
for a full `TenKOutput` on every per-field RAG call rather than the
single field being extracted.

**After #78 Option A (per-field with correct routing):**

| Cost line | Calls | Per-call | Subtotal |
|---|---|---|---|
| Narrative — 3 Haiku-routed fields | 3 | $0.015 | $0.045 |
| Narrative — 2 Sonnet-routed fields (+ thinking) | 2 | $0.045 + $0.031 = $0.076 | $0.152 |
| Item 8 (one prompt per table) | 5 | $0.006 | $0.030 |
| **Total** | | | **~$0.23** |

**Savings: ~44% per long 10-K.** Two effects compose:

1. **5× fan-out reduction** — each call emits one field, not all five.
2. **Tier correction** — 3 of 5 fields drop from Sonnet to Haiku
   (5× cheaper per input token, 3× cheaper per output token).

**After #79 complete financials + Option A:**

| Cost line | Calls | Per-call | Subtotal |
|---|---|---|---|
| Narrative (as above) | 5 mixed | mixed | $0.197 |
| Item 8 classifier per table | 5 | $0.002 | $0.010 |
| Item 8 extractor (3 statements) | 3 | $0.006 | $0.018 |
| **Total** | | | **~$0.23** |

Item 8 cost stays roughly flat. The classifier overhead is offset by
running only 3 extractors (one per actual statement type) vs 5 today
(one per raw table including notes-table noise).

### 4.2 Short 10-K (single-shot, <100K tokens)

| Path | Calls | Notes | Per-doc |
|---|---|---|---|
| Today | 1 narrative (Sonnet, wrong tier) + 1 Item 8 | All-in-one prompt | ~$0.08 |
| After Option A | 1 narrative + 1 Item 8 | Same call count; just better routing | ~$0.05 |

### 4.3 8-K

8-Ks are always short. Single Haiku call with closed-enum
classification + Claim lists.

**Per 8-K: ~$0.005.** Hardly affected by any of the in-flight work.

### 4.4 Transcript

A 25-60K-token earnings call transcribes into one Sonnet call today
(routed via `_TASK = q_and_a_evasiveness` because Q&A judgment is
the bottleneck per spec §7.3).

| Path | Calls | Per-call | Per-doc |
|---|---|---|---|
| Today (one fused Sonnet call) | 1 | $0.10 | ~$0.10 |
| After #6 split (prepared_remarks → Haiku; Q&A → Sonnet+thinking) | 2 | $0.015 + $0.045+$0.031 | ~$0.09 |

Savings is modest in absolute terms because the input dominates
(transcript text is large) and most of the cost is on the Sonnet half
that stays Sonnet. But the split unblocks per-prompt iteration
without invalidating the other half's cache.

---

## 5. Per-backfill-cycle math

### 5.1 Assumptions about the universe

Based on the spec's stated target and the current implementation:

- ~500 tickers, quarterly filing rhythm
- Per cycle (quarter):
  - ~500 10-Ks (~100 long + 400 short, rough split)
  - ~2000 8-Ks (~4 per ticker per quarter)
  - ~500 earnings transcripts (one per ticker per quarter)
  - ~10K contextual-chunking calls (varies with filing length)

These are guesses. The actual distribution is bimodal — FAANG 10-Ks
are 200K+ tokens; small caps are 30K. If the universe shifts toward
larger filings, total cost scales toward the long-10K column.

### 5.2 Total LLM cost per cycle

| Doc type | Count | Per-doc (today) | Per-doc (after Option A + #79) | Today | After |
|---|---|---|---|---|---|
| Long 10-K (RAG) | ~100 | $0.41 | $0.23 | $41 | $23 |
| Short 10-K (single-shot) | ~400 | $0.08 | $0.05 | $32 | $20 |
| 8-K | ~2000 | $0.005 | $0.005 | $10 | $10 |
| Transcript | ~500 | $0.10 | $0.09 | $50 | $45 |
| Contextual chunking | ~10K | $0.001 | $0.001 | $10 | $10 |
| **Total LLM cost per cycle** | | | | **~$143** | **~$108** |

### 5.3 Annual

Four cycles/year, plus eval (#20) overhead when it lands:

| | Per cycle | Per year (4 cycles) |
|---|---|---|
| Today (sync API, pre-follow-ups) | ~$143 | ~$570 |
| After Option A + #79 (sync API) | ~$108 | ~$430 |
| After follow-ups + Batch API | ~$54 | ~$215 |

**The headline number: extraction LLM spend is on the order of
$200-600/year at current universe scale.** That's the *entire* budget
envelope this cost model is sized for. Any infrastructure change
that costs more than this in engineering time has to justify itself
on grounds other than direct LLM-spend reduction.

---

## 6. Sensitivity analysis — what assumptions move the answer

The numbers above hide several uncertain assumptions. Each can shift
total cost by a factor of 2 or more:

### 6.1 Cache hit rate

All per-call numbers assume warm cache. If backfill runs are
*spaced out* (>5 min between calls per worker), cache misses dominate
and per-call cost roughly doubles.

**Mitigation:** the backfill orchestrator should batch per-worker
calls within 5-minute windows. This is already the natural pattern
when iterating a worker over its corpus; the failure mode is when
the orchestrator interleaves workers ("now do all Item 8 across all
filings, then all narratives across all filings, ..."), which kills
the per-worker cache locality.

### 6.2 Retrieved-content size

10K tokens per RAG call is the spec's top-5 reranked parents. If
retrieval returns top-10 (more recall, more context), per-call cost
roughly doubles on the input side. If top-3, halves.

This is a *quality* knob, not a cost knob — the right tradeoff
depends on eval signal. But it's the single biggest cost lever after
routing tier.

### 6.3 Output verbosity

Per-call output assumes ~1000 tokens of structured JSON. If a few-shot
example in the prompt encourages the model to echo similar verbosity
(which they often do), output grows. Extended thinking budget is
already accounted for on Sonnet calls; budgets above 2048 would add
linearly.

### 6.4 Filing length distribution

"100 long / 400 short 10-Ks" is a guess. The actual breakdown depends
on universe composition (large caps run longer filings). If the universe
shifts more long, the per-cycle total scales toward the long-10K column
($41/cycle for long 10-Ks vs $32/cycle for short).

### 6.5 Quarantine rate

Costs ignore retry from quarantine. If `_resolve_spans` quarantines
10% of outputs and we re-extract, multiply affected docs by ~1.10.
PR #77's correctness fixes should drive the AMBIGUOUS / INSUFFICIENT
quarantine rate near zero on production text; was likely much higher
pre-fix.

### 6.6 Eval suite cost (#20)

Running DeepEval / G-Eval against a holdout set adds **judge-LLM
calls** (typically Sonnet, sometimes Opus). For a holdout of ~50 docs
× 4 judge dimensions × Sonnet ≈ $5 per eval run. If eval runs nightly
during prompt iteration, that's a meaningful add (~$150/month) — but
it's a *transient* cost during the iterate-on-prompts phase, not a
recurring per-cycle cost.

---

## 7. What this model does NOT capture

### 7.1 Re-extractions when prompts/schemas iterate

Bump `EIGHT_K_PROMPT_VERSION` from v1 → v2 and the entire 8-K corpus
re-extracts the next time the backfill runs. Cost:
`corpus_size × per-doc cost = 2000 × $0.005 = $10`. Small for 8-K;
hurts more if it's a 10-K narrative bump (~$50 to re-extract all 500).

The orthogonal-cache-key discipline (per-field SCHEMA_VERSION in
Option A, per-statement SCHEMA_VERSION in #79) is what stops a bump
in one prompt from invalidating *every other* worker's cache.

### 7.2 The agent layer (issues #28, #29)

The research agent and live critic are Opus/Sonnet calls invoked
*per query*, not per backfill. They dominate cost when used but the
usage pattern is bursty/interactive, not throughput-based — different
budgeting axis entirely.

### 7.3 Failed-output reprocessing

Quarantined docs sit until human triage. If a fix-and-replay loop
kicks in (rare today; will be more common once a triage tool exists),
multiply affected docs by their per-doc cost.

### 7.4 Embeddings + reranking

Voyage embeddings (~$5/cycle) + Qwen3 reranker (local, free) are
separate cost categories that don't appear in the LLM model. They're
small relative to LLM cost today; could grow if the universe scales
significantly.

### 7.5 Infrastructure cost

LanceDB is local + free. MLflow tracking is local + free. OTel
emission to a managed backend (Langfuse, etc.) has its own
per-event pricing — typically dwarfed by LLM cost but worth knowing.

---

## 8. The single biggest lever: routing discipline

Look at the per-call cost table in §3.2. The same call shape
(10K input + 1K output) costs:

- $0.015 on Haiku
- $0.045 on Sonnet
- $0.225 on Opus (if we routed it there)

**That's a 3-15× spread purely from picking the wrong tier.**

In PR #77's pre-fix state, the 10-K RAG path routed everything to
Sonnet because `_NARRATIVE_DEFAULT_TASK = "supplier_mentions"` (a
Sonnet-tier task) was used for the unified call. The routing-table
rows for `(ten_k, guidance_tone) → Haiku`, etc., existed in
`_models.py` but never fired. That's what Option A unlocks.

**The lesson:** the routing table is a design artifact. If it's not
mechanically enforced by code, it silently drifts to "use the
strongest model by default" — which is the most expensive default
possible. Every PR that adds a new (worker, task) extraction should
add the corresponding routing-table row AND ensure the call site
passes the matching `task=` string.

---

## 9. Local inference — when does it pay back?

A perennial question: "why not run Qwen 32B locally and skip the
API entirely?"

### 9.1 Per-call cost comparison

| Backend | Per call (Item 8 extractor shape) | Notes |
|---|---|---|
| Haiku 4.5 (API, warm cache) | ~$0.006 | Today's baseline |
| Haiku 4.5 (API, Batch) | ~$0.003 | 50% Batch discount |
| Qwen 3.5 35B-A3B MoE (vLLM, H100 80GB rental) | ~$0.00003 | 200× cheaper than Haiku — MoE activates only ~3B per forward pass, so throughput is 5-10× higher than a 27-32B dense model on the same hardware |
| Qwen 3.5 27B dense (vLLM, H100 80GB rental) | ~$0.0001 | 60× cheaper than Haiku; predictable serving cost (no MoE routing variance) |
| Qwen 3.5 35B-A3B MoE (MLX, owned M3 Ultra) | ~$0.00001 | Hardware amortized over 3yr + electricity ~$0.10/kWh; MoE fits comfortably in 128GB unified memory |
| Qwen 3.5 9B (MLX, small Mac) | ~$0.000003 | Cheap-enough-to-be-free; surprisingly competitive on templated tasks (small-series benchmarks reportedly beat GPT-OSS-120B) |
| Llama 4 / equivalent 70B-class (vLLM, H100) | ~$0.0002 | Quality similar to Qwen 3.5 27B dense for our shape |

### 9.2 Per-cycle impact

| Backend | Cycle | Year (4 cycles) |
|---|---|---|
| Haiku API (current path) | ~$100 | ~$400 |
| Haiku via Batch | ~$50 | ~$200 |
| Local Qwen 3.5 35B-A3B MoE (cloud H100, peak only) | ~$2 | ~$8 |
| Local Qwen 3.5 27B dense (cloud H100, peak only) | ~$5 | ~$20 |
| Local Qwen 3.5 35B-A3B MoE (owned Mac M3 Ultra) | ~$0.20 | ~$1 |
| Local Qwen 3.5 9B (owned Mac, even smaller hardware) | ~$0.05 | ~$0.20 |

**Conclusion 1: the cost saving is real but trivially small in
absolute terms.** $200-400/yr down to $20-40. The economic case
alone doesn't justify the engineering investment at current scale.

### 9.3 Quality cleavage — where local works, where it doesn't

Haiku does very different things in this codebase. Local-model
viability differs sharply by task:

| Task | Haiku-eligible today | Local viable now? | Why |
|---|---|---|---|
| Item 8 line-item extraction | yes | **Yes** — Qwen 3.5 27B / 35B-A3B MoE match Haiku | Pure template matching; constrained generation enforces schema |
| Item 8 table classifier (#79) | yes | **Yes** — Qwen 3.5 9B is plenty | 7-class enum; trivial for any 9B+ model |
| 8-K event_classification (closed enum) | yes | **Yes** | Same reasoning |
| S-1/S-3 dilution_event | yes | **Yes** | Templated; sparse output |
| **Contextual chunking** (10K+ calls/cycle) | yes | **Yes — highest-volume Haiku worker** | Pure text rewrite; small in/out; Qwen 3.5 9B handles this comfortably |
| 10-K guidance_tone | yes (post-Option A) | **Maybe** — subjective | Local models often less-calibrated on tone; needs eval to confirm |
| 10-K accrual_flags | yes (post-Option A) | **Maybe** — needs accounting reasoning | False-positive risk; 27B+ class likely needed |
| 10-K risk_factor_deltas | yes (post-Option A) | **Risky** — needs prior-year comparison | Sonnet does this; Haiku already a stretch |
| Supplier/customer mentions (cross-doc) | NO (Sonnet) | **No** | Cross-doc reasoning; would regress |
| Q&A evasiveness | NO (Sonnet) | **No** | Subtle pragmatic judgment |

The clean cleavage: **templated extraction goes local, subjective
judgment stays on API**. About 60% of Haiku-eligible calls fall in
the "templated" bucket.

### 9.4 What you'd lose

1. **Anthropic prompt cache** — vLLM has its own prefix cache that
   auto-kicks-in; functionally similar.
2. **Tool_use's server-side schema validation** — replaced by
   SGLang/Outlines/Guidance client-side constrained generation.
   Equivalent guarantee, different code path.
3. **Extended thinking** — doesn't exist on most local models. Not
   a loss for templated tasks; relevant for the Sonnet-tier tasks
   we're keeping anyway.
4. **Citation discipline** — local models often paraphrase verbatim
   quotes more than Haiku does. `_resolve_spans` handles this fine,
   but quarantine rates may climb. Eval would tell.
5. **Future API improvements** — Anthropic ships better caching,
   longer context, better citations every quarter. Local OSS lags
   the frontier by 6-12 months on extraction quality.

### 9.5 Architectural fit

The codebase is already designed for this. The routing-table
abstraction (`_models.py:_ROUTING`) means adding e.g.
`_LOCAL_QWEN_35B_MOE` as a route is a one-line extension. The
`ExtractionFn` protocol abstracts the call. What you'd build:

1. **New route constants** in `_ROUTING` for the specific
   (worker, task) pairs you want to send local.
2. **`make_local_extraction_client`** parallel to `make_extraction_client`.
   Talks to vLLM HTTP API (or MLX in-process on Mac). Same Protocol shape.
3. **Dispatch in `_get_client`** (in #78 PR-C this consolidates to
   `_common.py`): `if model_id.startswith("local-"): return make_local_extraction_client(...)`.
4. **Cost-cap doesn't apply** to local — replace with rate-limit /
   queue depth.
5. **Reliability primitives still apply** — circuit-breaker for vLLM
   OOM crashes, retry for transient queue overflow.

Total infra build: ~3-5 days for serving + 1-2 days for dispatch
wiring. Plus ongoing operational burden (GPU mgmt, weight updates,
monitoring).

**MoE serving note.** Qwen 3.5's MoE variants (35B-A3B, 122B-A10B)
have a non-obvious infrastructure property: memory footprint is set
by *total* parameters (so 35B-A3B needs ~70GB at FP16, fits one
H100 or a 128GB Mac) but throughput is set by *active* parameters
(~3B for 35B-A3B). This means a single H100 serving 35B-A3B can
push throughput equivalent to a much smaller dense model while
maintaining quality of a much larger one. vLLM and SGLang both
support these MoE shapes out of the box as of mid-2026.

### 9.6 Decision framework — when to revisit

Don't move to local now. Trigger conditions for revisiting (any one):

- **Backfill scales 10×+** (universe grows from 500 → 5000 tickers,
  or cadence goes weekly). Then LLM cost climbs into $2,000-5,000/yr
  and the infra investment pays back in <12 months.
- **Eval (#20) lands AND shows local matches Haiku on the templated
  tasks** with no quality regression. Then "local for templated, API
  for judgment" becomes a real architectural decision rather than
  speculation.
- **Compliance constraint forces it** (extracting from a non-public
  corpus where data can't leave the network). Trumps cost.
- **Anthropic pricing changes adversely** or rate limits become
  operationally painful.

### 9.7 Lowest-risk first foothold (when the time comes)

**Contextual chunking** (`extract/prompts/contextual_chunk.py`):

- Highest-volume Haiku call (~10K calls per backfill cycle).
- Purely templated (output is a one-line context string).
- No schema adherence concern (free text).
- No citation grounding to break.

If a local model can handle anything in this pipeline, it can handle
this. A 1-week pilot project that delivers a real quality signal
AND ~30% cost reduction on the highest-volume LLM workload, with
minimal risk to the rest of the pipeline. **Qwen 3.5 9B (MLX on
existing Mac dev hardware)** is the right starting point — small
enough to run on commodity Apple Silicon, large enough to handle
templated rewrites comfortably. Expand to Item 8 line-item
extraction (Qwen 3.5 27B dense or 35B-A3B MoE) next if the pilot
succeeds.

### 9.8 A note on model-version drift

The specific model identifiers in this document (Qwen 3.5 27B, etc.)
are accurate as of May 2026. Open-source frontier moves fast — a
year from now the recommendation might be Qwen 4 or a successor
family. The durable content here is the **decision framework** (§9.6
trigger conditions) and the **architectural pattern** (routing-table
extension + `make_local_extraction_client`); the specific weights are
a snapshot. Re-evaluate the per-call costs in §9.1 against current
SOTA weights before acting on this section.

---

## 10. Worked scenario: Qwen 3.5 on Mac replacing Haiku

The general analysis in §9 doesn't tell you what to actually build.
This section walks through a concrete plan: run Qwen 3.5 locally on
Apple Silicon (matching the codebase's existing MLX stack for
Qwen3-Embedding) and migrate the templated Haiku workers to it.

> **Critique of the over-engineered version.** An earlier sketch of
> this section proposed a normalized `ExtractionResponse` +
> `UnifiedUsage` dataclass as the cross-provider abstraction —
> roughly the shape LiteLLM exposes. We chose against it. Reasoning:
>
> - **At our scale**, there is 1 provider in production today and 2
>   anticipated (Anthropic API + an OpenAI-compat local backend).
>   That's regime B. A normalized dataclass is right altitude for
>   regime D (N providers, frequent swaps, integration-framework
>   consumers); for regime B it is ~300 LoC of new abstraction +
>   shared types for portability benefits we won't use.
> - **Provider-specific richness lives inside each wrapper.**
>   Anthropic carries `cache_creation_input_tokens` (§3.4 canary),
>   extended-thinking budget, and `service_tier == "batch"` for the
>   batch discount. OpenAI-compat carries `finish_reason ==
>   "tool_calls"` and (eventually) prefix-cache metrics from vLLM.
>   None of these map cleanly to a shared dataclass; trying to fit
>   them either leaks Anthropic-isms into the OpenAI side or vice
>   versa.
> - **The smallest abstraction is the right one.** Each wrapper
>   returns `tuple[dict | str | None, UsageDict]` — provider-specific
>   parsing happens *inside* the wrapper; what crosses the boundary
>   is the parsed structured payload (when `output_schema=` is set)
>   or joined text + a tiny `UsageDict` (token counts + a couple of
>   `NotRequired` provider extensions: `cache_*`, `stop_reason`).
>   ~80 LoC of new code; trivially extensible if/when provider #3
>   arrives.
>
> If the codebase ever moves to a many-provider research-stack regime
> (Sonnet for cross-doc, Gemini for code, GPT-5 for grading,
> Llama-class for templated), revisit. Until then the discipline
> documented in `extract/client.py`'s docstring against LangChain
> applies equally to LiteLLM and bespoke normalization layers: stay
> at the lowest-altitude abstraction that solves the actual problem.

### 10.1 Hardware assumption

The codebase already uses MLX for embeddings (`extract/embeddings.py`
loads Qwen3-Embedding 0.6B/4B), so the dev/prod environment is Apple
Silicon by design. Three realistic configurations:

| Hardware | Memory | One-time cost | Models that fit | Best for |
|---|---|---|---|---|
| Mac Mini M4 Pro | 64 GB | ~$1,800 | Qwen 3.5 9B (FP16), Qwen 3.5 27B (4-bit) | Templated-only workers (contextual chunking, Item 8 line items, 8-K classification) |
| Mac Studio M3 Ultra | 128 GB | ~$5,000 | Qwen 3.5 27B dense (FP16), Qwen 3.5 35B-A3B MoE (FP16), Qwen 3.5 122B-A10B (4-bit) | Templated + light-judgment workers (guidance_tone, accrual_flags) |
| Mac Studio M3 Ultra | 512 GB | ~$10,000 | Anything in the Qwen 3.5 family at FP16 | Full Haiku replacement plus experimentation headroom |

For purely cost-driven Haiku replacement, the **Mac Mini M4 Pro at
64GB** is the right starting point — it fits Qwen 3.5 9B at full
precision and Qwen 3.5 27B at 4-bit quantization, which covers every
"templated" task in §9.3. If the codebase is already running on a
larger Mac (likely, given MLX usage), the marginal hardware cost is
zero.

### 10.2 Model assignment per worker

Concrete routing decisions based on §9.3's quality cleavage:

| Worker / task | Today's tier | Proposed local model | Justification |
|---|---|---|---|
| Contextual chunking (10K+ calls/cycle) | Haiku | **Qwen 3.5 9B** | Pure rewrite, no schema, output is one-line text. Highest-volume Haiku call by far — biggest savings unlock. |
| Item 8 classifier (#79) | Haiku | **Qwen 3.5 9B** | 7-class enum; tiny output. |
| Item 8 line-item extraction (#79) | Haiku | **Qwen 3.5 27B dense** OR **35B-A3B MoE** | Templated but needs precise label-to-value mapping; benefits from a slightly larger model. |
| 8-K event_classification | Haiku | **Qwen 3.5 27B** | Closed enum + Claim list; small enough for any 27B-class model. |
| S-1/S-3 dilution_event | Haiku | **Qwen 3.5 27B** | Templated; sparse output. |
| 10-K guidance_tone (post-Option A) | Haiku | **Qwen 3.5 27B** — gated on eval | Subjective; needs eval before flipping. |
| 10-K accrual_flags (post-Option A) | Haiku | **Qwen 3.5 35B-A3B MoE** — gated on eval | Needs accounting reasoning; MoE active params match Haiku quality. |
| 10-K risk_factor_deltas (post-Option A) | Haiku | **Keep Haiku** initially | Year-over-year comparison is harder; defer local migration until eval shows match. |
| Supplier/customer mentions | **Sonnet** | **Keep Sonnet** | Cross-doc reasoning; outside Haiku-replacement scope. |
| Q&A evasiveness | **Sonnet** | **Keep Sonnet** | Pragmatic judgment; outside scope. |

### 10.3 Per-call cost on Mac

The interesting cost calculation is that on a Mac you already own,
inference cost is dominated by hardware amortization, not per-call
compute. Working through the math:

**Assumptions:**
- Mac Mini M4 Pro 64GB, $1,800 hardware
- 3-year amortization → $50/month → ~$0.07/hour fully amortized
- Idle power ~10W, peak inference ~50W (Apple Silicon is very
  efficient)
- Electricity $0.10/kWh

**Per backfill cycle:**
- Total Haiku-eligible local calls (post-Option A + #79):
  - Contextual chunking: ~10K calls
  - Item 8 (classifier + extractor): ~8 per long 10-K × 100 long + ~2 per short 10-K × 400 = ~1600 calls
  - 8-K: ~2000 calls
  - S-1/S-3 (small volume): ~50 calls
  - 10-K templated narrative fields (guidance_tone, accrual_flags, risk_factor_deltas): ~3 per 10-K × 500 = ~1500 calls
  - **Total: ~15,000 local calls per cycle**
- Mean latency at Qwen 3.5 9B on M4 Pro: ~2 sec/call (generous;
  templated outputs are short)
- Mean latency at Qwen 3.5 27B: ~6 sec/call
- Time mix: 75% on 9B, 25% on 27B → mean 3 sec/call
- **Total compute time: ~12.5 hours per cycle**
- Electricity: 12.5h × 50W × $0.10/kWh = **~$0.06 per cycle**
- Hardware amortization attributable to backfill: 12.5h ÷ 720h/month
  × $50/month = **~$0.87 per cycle**
- **Total Mac cost per cycle: ~$0.93**

If the Mac is shared with the existing MLX embedding workload
(which it should be), allocate ~50% of hardware amortization to
extraction:

**~$0.50 per cycle**, vs ~$100 per cycle for the same workload on
Haiku API.

### 10.4 Per-cycle cost comparison

| Scenario | Per cycle | Per year | Savings vs current |
|---|---|---|---|
| All Haiku via API (today, pre-follow-ups) | ~$143 | ~$570 | — |
| All Haiku via API (after Option A + #79) | ~$108 | ~$430 | $140/yr |
| All Haiku via API + Batch (after follow-ups) | ~$54 | ~$215 | $355/yr |
| **Qwen 3.5 on Mac (templated only) + Haiku API for rest** | **~$8** | **~$32** | **$540/yr** |
| **Qwen 3.5 on Mac (all Haiku-eligible) + Sonnet API for cross-doc** | **~$3** | **~$12** | **$560/yr** |

(Both Mac scenarios add ~$0.50/cycle for the local compute itself.)

The headline: **Qwen 3.5 on Mac cuts the Haiku-eligible LLM bill from
~$100/cycle to ~$0.50/cycle.** Sonnet calls stay on the API (~$2-3
per cycle for cross-doc fields) and become the dominant LLM cost line.

### 10.5 Implementation steps

The codebase is already shaped for this — MLX is in the stack, the
routing table is the right extension point, the `ExtractionFn`
protocol abstracts the call. Sequenced rollout:

**Phase 1 — Infra (1 week):**

The architecture is `(text | dict | None, UsageDict)` tuple return from
the `ExtractionFn` Protocol — the same shape backs both the Anthropic
SDK wrapper (production today) and a new OpenAI-compatible HTTP
wrapper (local). Provider-specific parsing (Anthropic's `tool_use.input`
dict; OpenAI's `response_format=json_schema` JSON string) happens
inside each wrapper; what callers see is the structured payload (or
joined text when `output_schema=None`) plus a tiny `UsageDict` —
`input_tokens`, `output_tokens`, and `NotRequired` extensions for
`cache_*` (Anthropic only) and `stop_reason` (provider-raw). See the
callout above for why we picked this over a normalized
`ExtractionResponse` dataclass.

1. Add `_LOCAL_QWEN_4B`, `_LOCAL_QWEN_27B_DENSE`,
   `_LOCAL_QWEN_35B_MOE` constants in `auto_research/_models.py`
   alongside `_HAIKU`, `_SONNET`, `_OPUS`. The `local/` prefix is
   the dispatch hint that `_get_or_build_client` reads; the suffix
   is the server-native model id (HuggingFace repo path for
   vllm-mlx / mlx-openai-server; Ollama-tag form is acceptable for
   Ollama deploys). Only `_LOCAL_QWEN_35B_MOE` is smoke-tested; the
   other two are placeholders for follow-up tiers.
2. Change `ExtractionFn` Protocol return to
   `tuple[dict[str, Any] | str | None, UsageDict]`. Update
   `make_extraction_client._call` to parse `Message.content`
   (Anthropic `tool_use.input` for structured; joined text blocks
   for free-form) and lift `Message.usage` + `stop_reason` into
   `UsageDict`. The reliability decorators stay on a Message-returning
   inner so `@cost_cap` keeps reading `Message.usage` for per-call USD
   accounting.
3. Implement `make_openai_compat_extraction_client(...)` in
   `auto_research/extract/openai_compat_client.py`. Wraps
   `openai.OpenAI(base_url=..., api_key=...)`. Forwards
   `response_format={"type": "json_schema", ...}` when
   `output_schema=` is set (the OpenAI structured-outputs equivalent
   of the Anthropic path's `tool_choice=record_extraction`). Reliability
   composition is `circuit_breaker(retry(...))` only — no
   `@cost_cap` (local serving has no per-call $$). OpenAI-aware retry
   predicate retries 429 / 5xx / `APIConnectionError` and propagates
   4xx programmer errors.
4. Add dispatch in `workers/_common._get_or_build_client(worker,
   task, ...)`: `if route_model(worker, task).startswith("local/"):
   return get_or_build_local_client(...) else: make_extraction_client(...)`.
   The Anthropic singleton keys on `worker`; the local singleton
   keys on `(worker, model_id)` so a routing flip from 9B to 27B
   builds a fresh client.
5. Add OTel attributes on the local path: `llm.backend = "openai_compat"`,
   `llm.local_model_id` (server-native id after stripping `local/`),
   `llm.input_tokens`, `llm.output_tokens`. Cost (`llm.cost.est_usd`)
   is deliberately not emitted on this path because there isn't one.
6. **No `_ROUTING` rows flip to `local/*` in Phase 1.** A unit test
   pins this (`test_no_production_routes_flipped_to_local`); flipping
   a row requires removing the assertion AND citing eval evidence.

**Server choice (Ollama vs vllm-mlx vs other MLX servers).** The
wrapper is the OpenAI-compat HTTP shape, so all back-end choices use
the same client code; the decision is at deploy time, controlled by
`base_url`:

- **Ollama** (`:11434/v1`). Easiest DX — `ollama pull <tag>` and the
  server is up. Quantizes by default (Q4_K_M). Fine for prototyping;
  on Apple Silicon trails MLX by ~2× on decode and ~5× on prompt
  processing, and no client-visible prefix cache.
- **vllm-mlx** (`:8000/v1`). Native MLX backend with PagedAttention
  + prefix cache (the structural analogue of Anthropic's
  `cache_control: ephemeral` that §3.4's cache-read canary depends
  on). Right for Apple Silicon when the workload pattern is
  many-calls-share-a-system-prefix (the contextual chunker hits this
  pattern; per-field RAG narrative does too).
- **mlx-openai-server / MLX Omni Server** (custom port). MLX-native,
  simpler than vllm-mlx, no prefix cache on most builds. The
  drop-in if vllm-mlx's PagedAttention hits an edge case on a
  specific model.

### Locked stack (smoke-tested 2026-05-29 on Mac M2 96 GB)

For the M2-class development hardware this codebase deploys to, the
locked combination is:

- **Server**: `vllm-mlx==0.3.0` in a dedicated venv outside the
  project tree (avoids pulling vllm-mlx + its transformers / torch
  deps into the main lockfile).
- **Model**: `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` — Unsloth's
  dynamic-4-bit MLX checkpoint of Qwen 3.6-35B-A3B MoE. ~20 GB on
  disk, ~20 GB RSS at runtime. UD-quant preserves quality on
  sensitive layers (early/late attention, embeddings) at 6-8 bit
  while compressing FFN to 4-bit — empirically within 0.5-1.5
  points of FP16 on aggregate benchmarks, and well within the §10.6
  acceptable zone (<5 % citation-paraphrase quarantine).
- **Launch**: `make serve-local-llm` (or
  `./scripts/serve_local_llm.sh` directly). The script is the
  operator-facing source of truth for the launch flags; this doc
  references it and vice versa so the two don't drift. Override
  defaults via env vars (`MODEL=`, `PORT=`, `VLLM_MLX_VENV=`).
  Effective command:

  ```bash
  vllm-mlx serve unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit \
    --host 127.0.0.1 --port 8000 \
    --enable-prefix-cache \
    --max-tokens 4096 --max-request-tokens 16384 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
  ```

  Three of those flags are load-bearing and not the defaults — pin
  them on every relaunch:

  - `--enable-prefix-cache` — the cache-economics framing depends
    on it. Without it the contextual chunker re-processes the
    shared system prefix on every call, wasting most of the
    per-call latency budget.
  - `--max-request-tokens 16384` — sized for our actual workload
    (max ~10-15 K context on the RAG narrative path). The native
    256 K context would allocate a KV-cache reserve we never use
    and steal from the prefix-cache pool. Raise only if a new
    worker shape needs more.
  - `--default-chat-template-kwargs '{"enable_thinking": false}'`
    — Qwen 3.6 enables a "thinking mode" by default that emits
    prose-form chain-of-thought into `choices[0].message.content`
    (not into a separate `reasoning_content` channel). For our
    workload — structured JSON via `response_format=json_schema`,
    or short text rewrites for contextual chunking — CoT prose in
    `content` is junk that either trips the chunker's
    output-token cap or pollutes the structured payload. The
    `--reasoning-parser qwen3` flag is kept for the case where a
    future model does emit `<think>` tags; with `enable_thinking`
    off it is a no-op.

- **Wrapper config**: pass `base_url="http://127.0.0.1:8000/v1"` to
  `make_openai_compat_extraction_client`. No code change.

### Smoke-test results

Captured against the launch command above on M2 96 GB:

| Check | Result |
|---|---|
| Server `/v1/models` reachability | OK |
| OpenAI SDK free-form chat completion | clean text, `finish_reason=stop`, no CoT leak |
| OpenAI SDK `response_format=json_schema` (`strict: true`) | schema honored, JSON parsed cleanly |
| `make_openai_compat_extraction_client` structured path | `(dict, UsageDict)` round-trip; `usage["stop_reason"]="stop"` |
| `make_openai_compat_extraction_client` free-form path | `(str, UsageDict)` round-trip; non-empty text |
| Sustained decode (512 completion tokens) | **64.4 tok/s** |
| Process RSS at steady state | **20.2 GB** (leaves ~70 GB for KV cache + Qwen3-Embedding MLX workload) |

7. Verify the wrapper end-to-end against vllm-mlx serving the locked
   model: run a contextual-chunking call shape (free-form text, no
   schema) and an Item 8 classifier shape (structured,
   `response_format=json_schema`). Both should round-trip through
   the new Protocol without leaking provider-specific types. The
   smoke scripts at `tests/integration/` or ad-hoc `/tmp/`
   equivalents work; this is operator verification, not a
   committed test (the server isn't available in CI).

**Phase 2 — Pilot on contextual chunking (1 week):**

1. Flip ONE worker — contextual chunking — to `_LOCAL_QWEN_35B_MOE`
   in the routing table (the locked smoke-tested stack). Highest-volume
   Haiku call, lowest quality risk (free-text rewrite, no schema,
   no citations). At the same time, widen `_VALID_STOP_REASONS` in
   `chunking_contextual.py` to include OpenAI's `"stop"` synonym —
   the existing Anthropic-only allow-set would silently drop every
   local response.
2. Run a backfill cycle on a 10-ticker subset; compare:
   - Latency (per-chunk wall-clock)
   - Downstream retrieval quality (does the BGE/Voyage embedder
     produce comparable similarity scores on the local-contextualized
     chunks vs the Haiku-contextualized ones?)
   - Total cycle cost
3. If quality holds, expand to the full corpus on the next cycle.

**Phase 3 — Item 8 + 8-K + S-1 (after eval #20 lands, 2 weeks):**

1. With the eval suite providing per-task baselines, flip Item 8
   classifier + extractor to `_LOCAL_QWEN_35B_MOE` and 8-K + S-1
   workers to the same. The 27B-dense constant remains a fallback
   if the MoE's structured-output adherence on Item 8 line items
   regresses under eval.
2. Re-run eval against the local-extracted outputs; require parity
   on the citation-grounding and schema-adherence metrics.
3. Flip the routing table on a corpus subset; promote to full
   corpus once a full cycle passes eval.

**Phase 4 — 10-K narrative templated fields (after eval, 2 weeks):**

1. `guidance_tone`, `accrual_flags`: flip to `_LOCAL_QWEN_35B_MOE`
   (the locked stack handles both). These are the highest-risk
   migrations because they involve judgment; require strong eval
   signal before promoting.
2. `risk_factor_deltas`: defer indefinitely. Year-over-year delta
   reasoning is at the edge of what 30B-class models do reliably.

### 10.6 What can go wrong (and the mitigation)

| Risk | Mitigation |
|---|---|
| Local model paraphrases instead of verbatim-quoting | `_resolve_spans` AMBIGUOUS / INSUFFICIENT path catches this; quarantine rate is the signal. If rate > 5%, the local model isn't ready for that worker. |
| Schema adherence regresses (invalid JSON, wrong enum values) | Constrained generation (`outlines`, `mlx-lm`'s JSON mode) at the inference layer enforces schema. Falls back to quarantine on the rare miss. |
| Cold-start latency on first call | MLX caches loaded weights in unified memory; first-call ~5s, subsequent calls amortize to ms-level. Pre-warm at backfill start with a single throwaway call. |
| Mac runs hot under sustained load | Throughput is the relevant SLO, not latency. Backfill is throughput-bounded; let it run for hours. Mac thermal throttling is generous compared to typical GPU server. |
| Quality regression on accrual_flags / guidance_tone | Eval (#20) gate. Don't flip those routes until baselines pass. |
| MLX version churn breaks loading | Pin `mlx-lm` in `pyproject.toml`; treat the weight + library combo as a versioned artifact (same discipline as `EMBED_MODEL_VERSION_TAG`). |

### 10.7 The total story

A Mac you may already own + an open-source model + a 2-week
infra build + 4-6 weeks of phased rollout (gated on eval) gets the
extraction-pipeline LLM bill from ~$430/year to ~$12/year. That's
a 35× reduction.

**The unstated catch:** none of this is real until eval (#20) exists
to verify per-task quality parity. You can stand up the infrastructure
now (Phase 1), and the contextual-chunking pilot (Phase 2) doesn't
need eval because retrieval quality is independently measurable. But
Phases 3 and 4 are gated on eval — without it, the migration is
silent-degradation risk in service of saving $400/year, which is
exactly the wrong trade.

The right sequencing therefore looks like:

```
PR #77 (done) → #78 PR-A (tool_use) → #78 PR-B (Option A)
                ↓
                #20 (eval suite — UNLOCKS local migration)
                ↓
                Phase 1 infra → Phase 2 contextual chunking pilot
                ↓
                Phase 3 Item 8 / 8-K / S-1 → Phase 4 templated narrative
```

Local migration is downstream of eval. Eval is the unlock.

---

## 11. Patterns to internalize

If you remember nothing else from this document, remember these:

### 10.1 Routing is the biggest cost lever

A 3-15× cost spread between tiers means the single most important
thing to verify on any extraction PR is that the routing-table rows
fire as designed. Document the call site's `task=` string explicitly;
test that `route_model(worker, task)` returns the expected tier.

### 10.2 The cache TTL shapes the orchestration

Backfill orchestrators that interleave workers by document destroy
per-worker cache locality. Iterate by worker first, then by document,
to amortize the cache across the worker's full corpus pass.

### 10.3 Output is 5× input — design prompts accordingly

A prompt that asks for 3000 tokens of output costs the same as one
that asks for 200 tokens of output + 14000 tokens of input. The
Option A insight (5×full-output → 5×single-field-output) is worth
~40% cost on the RAG path because of this asymmetry. Verify your
prompts ask for the minimum output the schema requires.

### 10.4 Batch is 50% off, free engineering-wise

Any cost-reduction strategy should price-against-batch, not against
sync. The discount is automatic once the orchestrator hits the batch
API — no per-call code change. If the headline saving is "30% less
than sync," but you're sync-only because of a config issue, the real
saving is 0%.

### 10.5 Cache-hit visibility prevents silent regressions

The `llm.cache_read_input_tokens > 0` OTel attribute is the canary.
Without it, a refactor that strips `cache_control: ephemeral` would
triple per-call cost with no other signal. Every observability
attribute is paying for itself even when nothing is wrong, because
its job is to make wrongness *visible*.

### 10.6 Local-inference economics flip at 10× scale

The infrastructure-vs-API decision is non-linear in scale. At ~500
tickers, API wins on engineering effort. At ~5000 tickers, local
wins on absolute cost. At ~50000 tickers, local plus a multi-GPU
serving fleet is the only option. Know which regime you're in
before optimizing.

---

*Last updated: 2026-05-28. Pricing as of Anthropic's published rate
card on that date. Universe size, filing cadence, and call volume
assumptions reflect the current 500-ticker implementation; revisit
this document if those parameters change materially.*
