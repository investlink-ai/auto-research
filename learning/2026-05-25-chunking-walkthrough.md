# Chunking SEC filings — design walkthrough, lessons, and interview talking points

> Captured: 2026-05-25, from the Issue #13 work and its high-effort code review.
> This is teaching material — the *why* behind every decision, the categories of bugs the review caught, and the broader patterns to recognize in future work.

---

## 1. The problem and why it isn't trivial

### 1.1 What we're actually doing

We're parsing SEC HTML filings (10-K, 10-Q, 8-K, S-1, S-3) into **section-aware chunks** that downstream code can:

1. **Embed** for retrieval (RAG: BM25 + dense + reranker)
2. **Feed to an extractor** as context for typed Pydantic outputs
3. **Audit back to the source** when the extractor cites a claim

These three uses pull in different directions:

| Use | Wants chunks to be… |
|---|---|
| Retrieval | Small, semantically focused (~500 tokens) — embedding density |
| Extractor context | Large, with surrounding context (~4K tokens) — coherence |
| Audit | Slice-equivalent to the raw source — verifiability |

A single chunk size cannot satisfy all three. This tension drives the **parent / child two-level design** that came out of Anthropic's *Contextual Retrieval* paper (Sep 2024).

### 1.2 Why naive chunking doesn't work for SEC filings

Common defaults that fail:

- **Fixed-token windows** (LangChain's `RecursiveCharacterTextSplitter`). Splits arbitrarily mid-section. Item 1A bleeds into Item 7 in the same chunk. Cross-section content pollutes retrieval relevance.
- **Sentence splits**. iXBRL HTML has sentence-final punctuation buried inside `<span>` tags. Sentences span arbitrary depths of markup.
- **Markdown-style heading splits**. SEC filings don't use `<h1>` headers reliably; they use bold-styled `<div>` or `<span>` blocks. "Item 1A" is often rendered as styled prose, not a heading element.
- **Page splits**. 10-Ks ship as a single HTML document; the original PDF's page boundaries are not preserved.

The right answer for SEC filings is **structure-aware**: identify the Item sections (which are the actual semantic units the filer organized the document around) and chunk *within* those.

### 1.3 What makes this Tier 2

The chunking module touches **INV-2 (citation grounding)** — every extracted claim must have `source_text[span] == source_quote`. The chunker is upstream of extraction, so if chunk char_spans are wrong, every claim grounded against that chunk is also wrong.

That puts this module in the same risk class as `extract/guardrails.py` and `extract/schemas.py` — the *contract code* for the citation invariant. Tier 2 means: failing test first, named test evidence in PR body, Change Contract block.

---

## 2. The pipeline shape

Where chunking sits in the broader two-plane architecture:

```
EDGAR / IR audio   →   ingest   →   raw store
                                       │
                              parse_filing(html, metadata)
                                       │
                                       ▼
                           ┌────────  ChunkSet  ────────┐
                           │                            │
                       ParentChunks                ChildChunks
                       (≤ 4K tokens,              (200-800 tokens,
                        context unit)              retrieval unit)
                           │                            │
                           ▼                            ▼
                   extraction worker            embedding adapter
                   (single-shot OR              (Voyage finance-2,
                    RAG-augmented)               LanceDB per-doc +
                           │                     per-corpus index)
                           ▼                            │
                   typed Pydantic                       │
                   with source_span,                    ▼
                   source_quote                hybrid retrieve
                           │                   (BM25 + dense + RRF
                           ▼                    + bge-reranker-v2-m3)
                   citation guardrail                   │
                   asserts INV-2                        ▼
                           │                  feeds RAG-augmented
                           ▼                  extraction for big docs
                       Feast (PIT-baked)
```

Three downstream consumers of `ChunkSet`:

1. **Direct extraction** (small docs <100K tokens): worker takes the whole `ChunkSet` and feeds it to Claude single-shot
2. **RAG-augmented extraction** (big docs): children get embedded, retrieval returns top-k parents, parents go to Claude as context
3. **Structured table extraction**: Item 8 tables emit as `ParentChunk(table_html=<raw HTML>)`, 10-K worker reads `table_html` directly via pandas + a typed Pydantic financials schema, bypassing RAG entirely

The chunker doesn't know which path its output takes. That's deliberate — the chunker's contract is "produce a faithful, INV-2-compliant decomposition of this document," and consumers compose from there.

---

## 3. The chunking design

### 3.1 Parent / child levels (ADR D4)

The Anthropic 2024 *Contextual Retrieval* paper made two observations:

- **Small chunks embed better.** Sentence-level or paragraph-level chunks have concentrated semantic content; their vectors are more separable.
- **Large chunks read better for extraction.** A 200-token chunk has no surrounding paragraph; the extractor lacks the discourse context needed to interpret pronouns, references, definitions.

The standard fix is **parent-document retrieval**:

```python
@dataclass(frozen=True)
class ParentChunk:    # the context unit (≤4K tokens)
    text: str
    section_name: str
    char_span: tuple[int, int]
    token_count: int
    table_html: str | None
    metadata: ChunkMetadata

@dataclass(frozen=True)
class ChildChunk:     # the retrieval unit (200-800 tokens)
    text: str
    char_span: tuple[int, int]      # subset of parent's char_span
    token_count: int
    parent_id: str                   # back-reference
    section_name: str                # copied from parent
    from_table: bool                 # ADR D5
    metadata: ChunkMetadata
```

**Retrieval flow:** embed children → query matches a child → return the child's *parent* to the extractor. The extractor sees the larger context; the embedding index keeps its semantic clarity.

**INV-2 invariant:** child spans are strict subsets of parent spans. So `validate_char_spans` on both levels uses the *same* `source_text` (the raw HTML); both checks pass or both fail.

### 3.2 char_span as the load-bearing primitive

This is the single most important design decision. Every chunk's `text` is **literally** `html[char_span[0]:char_span[1]]`. INV-2 holds trivially because the chunk *is* the slice, by construction.

```python
def parse_filing(*, html: str, metadata: ChunkMetadata) -> ChunkSet:
    # ... section detection, packing, subdivision ...
    validate_char_spans(html, parents_list, children_list)  # defense-in-depth
    return ChunkSet(parents=tuple(parents_list), children=tuple(children_list))
```

Why this matters: when the downstream extractor cites `source_quote="...", source_span=(1234, 1567)`, the guardrail can verify `raw_doc[1234:1567] == source_quote` in O(1). No fuzzy matching, no whitespace-tolerant comparison, no LLM-judgment. The contract is byte-exact.

The cost: chunk text contains HTML tags (`<span>`, `<table>`, entities like `&#160;`). That's ~30-50% "noise" relative to clean prose. But:

- Claude reads HTML fine
- The token budget (4K parent, 800 child) is on the raw HTML, so chunks are bounded
- The benefit (INV-2 trivially holds) is enormous compared to the cost (slightly fewer "useful" tokens per chunk)

**This is the central design tradeoff.** A different team might choose decoded-text chunks with a separate raw-HTML pointer for audit. We chose the slice-equals-text path because the audit trail is the load-bearing invariant.

### 3.3 Section detection: regex on raw HTML, not unstructured's element model

The Issue brief said "library-first via `unstructured`". The implementation backed off from that.

**Why:** `unstructured.partition_html` classifies text blocks as `Title`, `NarrativeText`, `Table`, `ListItem`, etc. For SEC 10-Ks, an "Item 1A" header is classified inconsistently — sometimes `Title`, sometimes `Text`, sometimes buried inside a larger `NarrativeText` because the surrounding HTML wrapped it inside a `<p>`.

Real example from NVDA's 10-K:

```
NarrativeText: "We hold confidential, sensitive, personal and proprietary
information... Breaches of o Item 7. Management's Discussion and Analysis
of Financial Condit..."
```

The "Item 7" header is buried inside an unrelated paragraph. `isinstance(el, Title)` misses it entirely.

**The fix:** scan raw HTML directly with an entity-tolerant regex, then filter for "real" headers using two heuristics:

```python
_ITEM_HEADER = re.compile(
    r"\b(?i:item)(?:\s|&\#160;|&nbsp;|&\#x[Aa]0;)+(\d+[A-Za-z]?)\b",
)
# (?i:item) makes the keyword case-insensitive but leaves entity
# alternatives case-sensitive (HTML5 entities are case-sensitive)
```

Filter chain:

1. **Item number must be in `_VALID_10K_ITEMS`** (whitelist drops "Item 408 of Regulation S-K" rule refs)
2. **Must be preceded by a *block* tag** (`</div>`, `</p>`, `<h1>`, etc.) — inline `<span>` doesn't count, since `compared to <span>Item 7</span>` is a cross-reference, not a header
3. **Must have ≥200 alpha chars of prose in the next 2KB**, starting 200 bytes past the candidate (skips the header's own text to avoid TOC echoes where adjacent Item names contribute density)
4. **Comment-masked**: matches inside `<!-- ... -->` are stripped (e.g., fixture truncation markers)

This is a **classic structure detection problem**: the document follows a logical schema (SEC Form 10-K), but the rendering is variable. The regex catches the schema; the heuristics filter the noise.

### 3.4 The table-handling decision (ADR D5)

10-K Item 8 is *the financial statements* — 30-50% of the doc's tokens, ~100% tabular. Three reasons text-based RAG fails on tables:

1. **Dense indexing is bad for tables.** A table embedded as a single chunk dilutes signal across 100+ rows. A table split row-by-row loses table structure.
2. **The extractor wants structured data.** Income statement, balance sheet, cash flow — these are typed schemas (revenue, COGS, EBITDA, etc.). Text-based extraction is roundabout when the source is structured.
3. **HTML well-formedness is fragile.** Splitting a `<table>` at `</td>` boundaries produces fragments without `</table>` — invalid HTML for any downstream consumer.

**The policy:** tables get a special path.

```python
@dataclass(frozen=True)
class ParentChunk:
    ...
    table_html: str | None  # raw <table>...</table> when this chunk IS a table
```

- `<table>...</table>` regions inside a section emit as `ParentChunk(text=<raw>, table_html=<raw>, char_span=<bounds>)`
- The 10-K worker reads `table_html` directly via `pandas.read_html(io.StringIO(t.table_html))` and a typed Pydantic schema
- Dense retrieval ignores chunks where `table_html is not None` (filter at the child level via `from_table=True`)
- Table parents are **atomic** — `subdivide_to_children` short-circuits to a single child equal to the parent
- Nested `<table>` depth is tracked so the outer table's `table_html` is never truncated at an inner `</table>`

This is **dual-path extraction**: narrative goes through RAG → Claude → typed text; tables go through pandas → typed numerics. Same document, two paths, both INV-2-compliant.

---

## 4. INV-2 in depth

This is the invariant that defines the module's correctness. Worth understanding deeply.

### 4.1 What it actually says

> Every extracted claim is citation-grounded. Pydantic schemas in `src/auto_research/extract/schemas.py` carry `source_span: tuple[int, int]` and `source_quote: str` on every claim. Post-validation asserts `source_text[span[0]:span[1]] == source_quote`. Failures route to `data/quarantine/` — never silently retried with degraded data.

The contract has **three parts**:

1. **Schema-level**: every claim CARRIES the span + quote (enforced by Pydantic at construction time)
2. **Runtime-level**: post-validation ASSERTS the slice equality
3. **Routing-level**: failures QUARANTINE, never silently degrade

### 4.2 Why all three matter

Skipping any one:

| Skip | Failure mode |
|---|---|
| Schema | LLM omits source_quote on some claims; "best effort" extractions leak through |
| Runtime | LLM hallucinates spans (it's known to be bad at counting chars); broken claims look valid |
| Routing | A try/except logs the error and returns the corrupted output anyway; bad data lands in Feast |

The routing piece is the **subtlest**. It's tempting to write:

```python
try:
    output = validate_citation_grounding(claims, source)
except CitationMismatch as e:
    logger.warning("citation mismatch", error=e)
    return claims  # ← the silent-degradation path
```

The `guardrails.py` module *explicitly forbids* this pattern via:

- A typed `CitationMismatch` exception (so callers can't catch generic `ValueError` and shrug)
- A `validate_or_quarantine(claims, source_text, *, doc_id, worker) → claims | None` wrapper that handles routing internally
- A meta-test (`test_no_disabling_flags_on_public_api`) that scans the module's public surface for `permissive`/`soft_mode`/`skip_validation` kwargs and fails the suite if any appear

The chunking module had to **mirror the same discipline** — typed `ChunkValidationError`, `validate_or_quarantine_chunkset` wrapper, meta-test extension. The review caught this: the wrapper was missing in the first version, and the meta-test only covered guardrails.

### 4.3 How chunking is upstream of INV-2

The chunker produces `char_span` values that downstream code uses to compute the slice. If chunking violates INV-2 — e.g., emits a chunk whose `text` is the decoded form (`Item 1A`) while `char_span` points at the encoded form (`Item&#160;1A`) — the extractor's downstream slice equality check would fail on a claim that's actually correct.

So the chunker enforces its own version of INV-2:

```python
def validate_char_spans(
    source_text: str,
    parents: Iterable[ParentChunk],
    children: Iterable[ChildChunk],
) -> None:
    for p in parents:
        a, b = p.char_span
        if source_text[a:b] != p.text:
            raise ChunkValidationError(...)
    # ... same for children
```

The chunker's char_span IS the extractor's source_span (modulo offset arithmetic for sub-chunk citations). The two halves of INV-2 share a single coordinate system.

---

## 5. The 16 review findings — as a learning catalog

The high-recall code review (plus Codex's automated pass) surfaced 16 issues across 5 reviewer angles. They cluster into categories that show up in *every* nontrivial codebase. Worth recognizing the categories, not just the specific bugs.

### Category A: discipline drift from established patterns

**Missing routing wrapper**: `parse_filing` raised on validation failure but had no single-call wrapper like `guardrails.validate_or_quarantine`. Callers had to remember to manually try/except + call `quarantine_chunkset`. The discipline gap means a future worker could write `chunkset = parse_filing(...)` and have the exception bubble up as an uncaught crash, leaving no quarantine record.

**Lesson:** when a parallel subsystem exists with an established contract, *mirror its shape exactly*. A new INV-2 entry point that "almost" matches the existing one is worse than a new entry point that mirrors it completely.

**Meta-test gap**: the "no permissive flags" test only scanned `guardrails`, not `chunking`. A future `parse_filing(..., permissive=True)` would land silently.

**Lesson:** meta-tests need to enumerate over their actual surface, not the original surface they were written against. When you split or extend an invariant across modules, scan all the modules.

### Category B: type-system lies

**`ChunkSet` frozen with mutable lists**: `@dataclass(frozen=True)` prevents *rebinding* the field (`chunkset.parents = []`) but not *mutation* (`chunkset.parents.append(...)`). Future consumers could silently corrupt the chunkset post-validation.

**Lesson:** "frozen" in Python is structural-only. To get value-level immutability, the field types themselves must be immutable — `tuple[X, ...]` instead of `list[X]`.

**`ChildChunk` lacks `section_name`**: downstream code (LanceDB schema) needed `section_name` for index-time filtering. Without it, the consumer would have to JOIN children against parents to recover it — an undocumented denormalization step the producer should have handled.

**Lesson:** types are downstream contracts. When designing a producer, walk through every consumer's needs and bake them into the dataclass. The cost is small (one field); the cost of retrofitting is huge (schema migration on the index).

### Category C: heuristics that work on the happy path

**`_looks_like_block_header` accepts inline `<span>`**: the original regex matched ANY preceding opening tag. Real SEC docs include `<span>Item 7</span>` mid-prose as cross-references. The fix restricted to block-level tags only.

**`_is_real_section_header` double-counts the header**: the 200-alpha-char threshold ran over a 2KB window starting at the candidate. The candidate's own text ("Item 1A. Risk Factors") contributed ~30 alpha chars. For dense TOC blocks where multiple Item names sit in 2KB of nav text, the threshold passed trivially. The fix skips the first 200 bytes.

**Lesson:** structure-detection heuristics are *adversarial* against the data — every shape you didn't anticipate will appear. Write the heuristic, run it on real data, look at its mistakes, refine. Two iterations is the minimum.

### Category D: regex flags that over-match

**`re.IGNORECASE` over the entity alternatives**: `_ITEM_HEADER` matched `Item&NBSP;5` (uppercase entity name). HTML5 entity names are case-sensitive; browsers reject `&NBSP;`. The over-match created phantom sections inside fuzzed/malformed input. Fix: `(?i:item)` localizes case-insensitivity to the keyword.

**Lesson:** regex flags are global to the pattern. When you have mixed-sensitivity needs (case-insensitive keyword + case-sensitive HTML entities), use scoped inline flags `(?i:...)` rather than a top-level `re.IGNORECASE`.

### Category E: silent overwrites and identity collisions

**Quarantine `doc_id="empty"` collision**: when both parents and children are empty, `quarantine_chunkset` filed at `chunking/empty.json`. Two empty quarantine events overwrote each other via `atomic_write_text`. Audit trail lost.

**Lesson:** any time you derive a path from input data, check what happens when the input is degenerate. The fix was forcing the caller to pass `doc_id` explicitly — make the producer name the artifact.

### Category F: nested-structure bugs

**`</table>` depth**: SEC iXBRL filings sometimes wrap inner financial-statement tables inside an outer layout table. The original code took the first `</table>` after each `<table>`, truncating the outer table at the inner close.

Fix: depth-counter walk.

```python
def _find_matching_table_close(html: str, after: int) -> int | None:
    depth = 1
    pos = after
    while True:
        open_m = _TABLE_OPEN.search(html, pos)
        close_m = _TABLE_CLOSE.search(html, pos)
        if close_m is None: return None
        if open_m is not None and open_m.start() < close_m.start():
            depth += 1
            pos = open_m.end()
        else:
            depth -= 1
            if depth == 0: return close_m.end()
            pos = close_m.end()
```

**Lesson:** any time you're matching paired delimiters with `find_first_close`, ask whether the structure can nest. HTML, JSON, parentheses, comments — they all can. Depth-counter walks are the standard fix.

### Category G: import-time side effects

**`_ensure_nlp_warmup()` at module import**: the original code called the spaCy model loader at module load time. Side effects:

- Importing the module added ~2s to import time
- mypy/ruff/IDE plugins that import the module needed the model installed
- Pytest collection could fail before any test ran
- Any future transitive importer inherited the dependency

Fix: lazy via `_NLP_WARMED` flag, called inside `parse_filing` on first use. Conftest fixture warms it once per test session before the no-network test.

**Lesson:** module imports should be *cheap*. State side-effects belong to first-call lazy initialization, not import. The cost of getting this wrong scales with the number of transitive importers.

---

## 6. Broader patterns this exemplifies

### 6.1 The two-plane architecture

The single most important architectural decision in this codebase is that the LLM is **never** in the trading-decision path.

```
LLM plane (async, batch)              Deterministic plane (synchronous)
─────────────────────────              ───────────────────────────────────
extraction workers                     Feast (PIT-baked features)
research agent                         signal library
live critic (haircut only)             backtest engine (CPCV, costs)
                                       paper portfolio
            │                                        ▲
            └── typed claims with source_span ───────┘
                Feast is the ONLY cross-plane contract
```

Why this matters:

- LLMs are non-deterministic. Trading decisions need reproducibility for audit, debugging, regulatory questions.
- LLMs are slow. Trading loops need to clear in microseconds for some venues, milliseconds for others.
- LLMs are expensive per call. Backtests run millions of decision points.
- LLMs hallucinate. Hallucinated trading signals are catastrophic; hallucinated feature values can be quarantined.

**The LLM's job is feature extraction from unstructured text — the one place it's uniquely valuable and where its non-determinism can be controlled (cache, version-pin, audit, quarantine).**

This is the *architectural corrective* to the popular `virattt/ai-hedge-fund` design which puts the LLM at the trading decision layer. That design fails for every reason above.

### 6.2 Citation grounding as research discipline

The INV-2 contract — every claim must have a verbatim source quote that slices back to the source — is what makes the system **defensible under interview probing**.

Interviewer: "What if the LLM extracted a fact that wasn't actually in the document?"

You: "Two things prevent it. First, the Pydantic schema requires every claim to carry `source_quote` and `source_span`. The model can't produce a valid output without naming the slice. Second, post-validation asserts the slice equality — if the LLM made up a quote, the byte-level slice comparison fails and the entire extraction routes to quarantine. The quarantine record carries the offending parsed output unmutated so a human reviewer can see what the model actually said. No silent retry, no permissive flag."

That answer is *mechanical*, not a hope. The interviewer can check the code: `validate_citation_grounding` in `guardrails.py`, post-validation in every worker, the meta-test that scans for forbidden kwargs.

This is what López de Prado calls "research that survives contact with reality" — discipline baked into the data flow, not relying on best efforts.

### 6.3 Parent-document retrieval as a published technique

The two-level chunking design is **not invented here**. It's the textbook fix for the small-vs-large chunk tradeoff. Key references:

- **Anthropic, *Contextual Retrieval* (Sept 2024)** — introduced "contextual chunking" (a one-line LLM-generated context prepended to each chunk before embedding). ~50% retrieval-recall lift on their benchmarks.
- **LlamaIndex / LangChain parent-document retrieval** — same pattern, different implementations.
- **OpenAI Assistants API file_search** — uses something similar internally.

Knowing the literature is what makes you sound like someone who's done this before. In interviews:

> "The two-chunk-size design comes from the Anthropic contextual-retrieval paper from September 2024. Small chunks embed better — sentence-level vectors are more separable. Large chunks read better — the extractor needs surrounding paragraph context. Parent-document retrieval gives you both: embed small children, return their parents to the consumer. We keep INV-2 holding by making child spans strict subsets of parent spans, so the slice-equality check operates identically at both levels."

### 6.4 Library-first, with judgment

The repo's `AI_CODE_STYLE.md` §4 says "library-first" — check for a reliable SDK before writing custom logic. The chunking module **imports** `unstructured` but **does not use** its output for chunking. The decision was:

- `unstructured.partition.html` is a 2GB-deep dependency (NLTK + spaCy + lxml + many tokenizers)
- Its element classification is unreliable for SEC HTML (verified empirically)
- Section detection via raw-HTML regex is more reliable for SEC docs
- But: keeping `unstructured` as a transitive dep means downstream extractors can still use its output if helpful (e.g., for non-SEC inputs)

This is **judgment, not dogma**. "Library-first" doesn't mean "library at all costs." It means "default to the library; if you can articulate a specific reason it doesn't fit, write the custom code and document the reason." The ADR D1 amendment recording this departure is the artifact of the judgment.

### 6.5 Industrial-best-practices documentation

ARCHITECTURE.md as it stands now uses:

- **C4-style layering** (Context → Container → Component)
- **Mermaid diagrams** in markdown (rendered inline on GitHub, version-controlled)
- **Time-invariant content** (no file-tree maps that rot)
- **Cross-references** between docs (CONTRACTS ↔ DATA_MODEL) so each file owns its primary concern

C4 is Simon Brown's model from 2018. It's the de facto standard for system-design docs at companies that take architecture seriously (especially in the Java/.NET enterprise world; less common in Python startups, more common in regulated/quant shops). The four levels:

1. **Context** — system + external actors. Who uses it, who feeds it.
2. **Container** — major executable boundaries (processes, services, stores).
3. **Component** — logical decomposition within a container.
4. **Code** — class-level (rarely worth drawing).

For interviews: "I documented the system using a loose C4 model with Mermaid diagrams. The motivation was to make the architecture time-invariant — folder layouts change every sprint, but the conceptual containers (ingest, extract, store, signal, backtest, agent) are stable for the life of the project. Reviewers can read the architecture once and not have to re-orient when files move."

---

## 7. Interview angles — what to bring up

### 7.1 The architectural-judgment story

**Hook:** "I rewrote one of the most popular open-source LLM trading projects. The original puts the LLM at the trading-decision layer; mine puts it at the feature-extraction layer."

**Tradeoff articulation:**
- LLM at decision layer: non-deterministic, slow, expensive per call, hallucinates trading directions
- LLM at extraction layer: each call has typed output, content-hash cached, citation-grounded, quarantined on failure
- Feast feature store is the only contract between the two planes — no LLM output bypasses PIT discipline

**Defensible specifics:** INV-1 (PIT lag-1 baked at write-time), INV-2 (citation grounding with quarantine routing), INV-3 (LLM never in trading-decision path, multiplicative haircut only). All seven invariants are mechanical, not aspirational.

### 7.2 The research-discipline story

**Hook:** "How do you keep an LLM-driven research agent honest?"

**Answer:**
- Every claim it extracts carries a verbatim source_quote and a span; post-validation asserts the slice. Failures route to `data/quarantine/`, never silent retry.
- Promote/iterate/kill decisions are code-checked constants (`T1_GATE`, `T2_GATE`), not LLM judgments.
- Backtest discipline is López de Prado: CPCV with embargo, triple-barrier labels, deflated Sharpe across all hypotheses tested.
- The research agent's job is to *propose* and *document*; the gates *decide*.

This is the "research that survives contact with reality" framing — discipline baked into the data flow, not best-effort.

### 7.3 The cost-engineering story

**Hook:** "Walk me through your unit-economics analysis on this project."

**Answer:**
- 2-year backfill of ~2,700 docs across ~90 tickers
- Anthropic API ~$100-200 (Batch API 50% discount + prompt caching at $0.30/MTok)
- Whisper for transcripts ~$10/mo + $130 one-time
- Voyage embeddings ~$5
- **Total external API spend: ~$250 for the whole 2-year backfill + 4-week dev cycle**
- This forced specific decisions: unstructured OSS lib instead of $0.03/page SaaS ($8K out of budget); voyage-finance-2 not voyage-3-large (same cost tier, financial-domain tuned); local BGE reranker not Cohere ($0 vs paid).

Knowing the numbers cold is the difference between "I built an AI thing" and "I built it within a $250 budget."

### 7.4 The code-review story

**Hook:** "How do you do code review?"

**Answer:**
- Multi-angle, high-recall pass for complex changes. The chunking module had five independent reviewer angles run in parallel: line-by-line, removed-behavior auditor, cross-file tracer, Python-pitfall specialist, wrapper/proxy correctness.
- Findings classified P0/P1/P2 with explicit failure scenarios for each.
- Codex (GitHub's automated reviewer) catches one category (nested-structure bugs that humans skim over).
- All 16 findings fixed before merge, with named test evidence for each.

The point of articulating this in an interview: it shows you treat code review as a *system*, not a vibe.

### 7.5 What an interviewer might probe

| Probe | Sketch of answer |
|---|---|
| "Why not use LangChain's chunker?" | Section-blind for SEC docs; cuts mid-Item; no char_span fidelity for INV-2. |
| "Why not the unstructured hosted API?" | $0.03/page × 270K pages ≈ $8K, over budget. OSS library does the same job locally. |
| "What if the LLM hallucinates a source quote?" | Post-validation slice equality catches it byte-exact; quarantine routes the failure with the unmutated parsed output for audit. |
| "What's the cost of getting INV-2 wrong?" | Silent degradation — extraction looks valid but cites non-existent text. The Feast store accumulates corrupted features; signals built on it produce false alpha. Quarantine is the firewall. |
| "What's `from_table` for?" | Hybrid retrieval needs to skip table-fragment children at query time, not after re-resolving to parents. It's a child-level filter for ADR D5's table-policy contract. |
| "Why parent/child rather than just one size?" | Anthropic contextual-retrieval paper: small chunks embed better, large chunks extract better. Two-level resolves the tradeoff. |
| "Why tuples in ChunkSet?" | Frozen dataclasses prevent rebinding but not mutation; tuple fields extend immutability to the contents. Closes a real bug we caught in review. |
| "What's the threat model for char_span?" | Adversarial HTML that exploits parser-normalization to make the chunker's char_span point at different text than the consumer's parse. Mitigated by keeping char_span on the raw input — both ends compute slices against identical bytes. |

### 7.6 Things you should NOT bring up

- "I used Claude Code / Cursor to write this." (Yes, fine, everyone does. Doesn't differentiate.)
- "I followed the README." (Doesn't show judgment.)
- The line counts. (Doesn't matter.)
- The fact that it works. (Bare minimum.)

**Things you SHOULD bring up:**

- Specific invariants and their mechanical enforcement
- The tradeoffs you considered and rejected (and why)
- The bugs you caught in review and the categories they belong to
- The literature you drew on (Anthropic contextual retrieval, López de Prado, Loughran-McDonald)
- The cost-driven decisions

---

## 8. What I would do differently with hindsight

A list of things the work would benefit from, if there were budget for v1.5:

1. **A property-based test for char_span fidelity.** Hypothesis-generated HTML with random entities, nested tags, edge cases → assert `validate_char_spans` always holds or `parse_filing` raises. Catches the bug class earlier.

2. **An actual `text_as_html` table-extraction fallback.** Right now if `pandas.read_html(table_html)` fails, the worker has no Plan B. A second pass via `lxml.html.fragments_fromstring` would catch malformed but recoverable tables.

3. **A separate per-corpus index for Signal A1.** The ADR D11 specifies this but it's downstream; the chunker doesn't write it. A future change could have chunking emit a corpus key on each chunk so the embedding writer doesn't have to derive it.

4. **Eval-driven calibration of the `_is_real_section_header` threshold.** The 200-alpha-char threshold is a heuristic. Run it against 50+ real 10-Ks, count false positives/negatives, tune. Until that data exists, the threshold is best-guess.

5. **More aggressive comment masking.** Right now `_mask_comments` handles only `<!-- ... -->`. If the input has nested comments (rare but possible), the masking is incomplete because Python's `re` doesn't support recursive patterns. A real HTML parser (`html.parser.HTMLParser`) would be more robust.

6. **Document `validate_or_quarantine_chunkset` as the *only* entry point.** Right now `parse_filing` is also public; future workers might call it directly and bypass routing. A meta-test could check that worker code never calls `parse_filing` without going through the wrapper.

7. **Real-time stability check on `unstructured` version pin.** The pinned version (0.21.5) was chosen for wrapt-compat with langfuse v2. If langfuse goes v3, the pin can move forward but the chunker may need re-validation. A nightly CI job that runs the chunking suite against `unstructured`'s latest release would catch drift.

These are **stretch items**, not v1 blockers. The work is mergeable as-is; these would harden it for v1.5.

---

## 9. Recommended reading

For deepening understanding of the patterns this exemplifies:

| Topic | Source |
|---|---|
| Contextual retrieval | Anthropic, "Introducing Contextual Retrieval" (Sept 2024) |
| Citation-grounded LLM extraction | Anthropic, "Citations" docs; Pydantic AI source-grounding patterns |
| Backtest discipline (CPCV, deflated Sharpe) | López de Prado, *Advances in Financial Machine Learning* (Ch 7, 13, 14, 15) |
| Financial-text language features | Loughran & McDonald, "When Is a Liability Not a Liability?" (JF 2011); Tetlock, "Giving Content to Investor Sentiment" (JF 2007) |
| Architectural documentation | Simon Brown, *The C4 Model for Visualizing Software Architecture* (free at c4model.com) |
| Industrial code review | Google Engineering Practices, "How to do a code review" |
| Two-plane LLM architectures | Karpathy's "LLM OS" lecture; the broader "LLM-as-feature-extractor" pattern in industrial RAG |

---

## 10. The 30-second pitch

If asked "what did you build?":

> A two-plane multi-agent platform for cross-asset language-driven alpha. The LLM extracts typed claims from SEC filings and earnings transcripts; every claim carries a verbatim source quote and a byte-span, and post-validation enforces slice equality before anything reaches the feature store. The trading side is fully deterministic — Feast for PIT discipline, López de Prado backtest gauntlet (CPCV, triple-barrier, deflated Sharpe), real cost plumbing into vbt.pro. The LLM never sits in the trading-decision path; promote/iterate/kill is code-checked against constants. The architectural corrective to the popular `ai-hedge-fund` repos that put the LLM at the wrong layer.

That's the headline. Everything in this walkthrough is one layer of detail below it. In the right interview you might never get past the pitch; in a deeper one, you can recurse into any subtree (chunking, INV-2, the review process, the cost model) and have a defensible answer.
