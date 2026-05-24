# ADR: RAG layer enhancements — embeddings, reranker, parent-document retrieval, table handling

- Date: 2026-05-24
- Status: **Accepted**
- Owners: Sam

## Context

Picking up Issue #13 (`feat(extract): unstructured.io parsing +
section-aware chunking`) prompted an end-to-end review of the RAG
pipeline as specified in `docs/specs/2026-05-22-design.md` §8.1–§8.3.
Eleven candidate enhancements surfaced, grouped into four concerns.

### 1. Model selection in §8.1/§8.2/§8.3 is generic, not financial

The spec names `voyage-3` for embeddings (`design.md:201,210,219`) and
`bge-reranker-base` for reranking (`design.md:212`). Both predate the
domain-tuned alternatives:

- **`voyage-finance-2`** — Voyage AI's SEC-filings + earnings-news tuned
  model. Same API surface, same `$0.12/1M tokens` price tier as
  `voyage-3`. Voyage's published benchmarks report ~5–15% retrieval lift
  on financial documents vs. the generic family.
- **`bge-reranker-v2-m3`** — same size class as `bge-reranker-base`,
  same CPU footprint, materially better recall on long passages. Drop-in
  via `sentence-transformers`.

Backfill embedding cost at our volume (~400K chunks × ~500 tokens ≈ 200M
tokens × $0.12/1M) is **~$24** either way — within the §3 ~$250 total
external-API budget for both models. No cost tradeoff.

### 2. The `Chunk` dataclass lacks metadata needed by signal A1

Issue #13's spec lists `text, section_name, char_span, token_count`.
Signal A1 (§9.1 — cross-doc forward-tone propagation) requires retrieval
windowed by **ticker** and **filing_date**: "time-decayed-weighted sum
of mentions of T … extracted from narrative-source docs in trailing 60
days." Without `(ticker, filing_date, fiscal_period, doc_type)` carried
on each chunk → embedded as LanceDB columns → filtered at index time,
the cross-doc retrieval either over-retrieves (relying on the reranker
to clean up) or fans out per-doc queries (slow, doesn't merge globally).

The fix is cheap **before** the chunks are materialized; expensive
after, because the LanceDB schema would need a migration and the
embedded vectors would all need rewrite to carry the new columns.

### 3. "One chunk size both for retrieval and for context" is a known suboptimality

The spec implies a single 4K-token-max chunk used as both the retrieval
unit (embedded into LanceDB) and the context unit (handed to the
extractor). Large chunks (~4K) embed poorly — semantic signal is
diluted across multiple topics — while small chunks lack context for
extraction. **Parent-document retrieval** (small child chunks for
search, large parent chunks for context) is the standard fix and lifts
retrieval recall ~15–30% in published benchmarks.

INV-2 (citation grounding) is preserved: child `char_span`s are subsets
of parent `char_span`s; the post-validation check works identically on
either granularity.

### 4. 10-K Item 8 is tabular; pure text RAG fails on it

`unstructured` returns `Table` elements distinctly from `NarrativeText`.
10-K Item 8 (audited financial statements) is 30–50% of doc tokens and
~100% tabular. Text-embedded tables dilute retrieval signal and confuse
extractors that expect prose. The spec doesn't say how tables are
routed.

### 5. Remaining concerns surfaced but lower-priority

- The `100K`-token cutoff between single-shot and chunked extraction
  (`design.md:195–196`) is a round number, not a measured threshold.
  Claude Sonnet/Opus context is 200K; effective single-shot with caching
  is ~190K.
- Default RRF weighting (k=60, equal weight BM25 / dense) is fine for
  v1 but SEC docs have query types — ticker symbols, dollar amounts —
  where BM25 should dominate.
- The contextual-chunking prompt (Issue #14) is subject to INV-6: the
  cache key must include the prompt version, or edits silently keep
  stale contexts forever.
- `unstructured.partition_html` does not guarantee byte-exact offsets
  through HTML normalization. The INV-2 char_span contract requires
  explicit verification.
- LanceDB per-doc stores (`design.md:202`) work for per-doc extraction
  but Signal A1's cross-doc retrieval needs either fan-out or a parallel
  per-corpus index.

## Decision

Eleven enhancements, grouped as before. Five are **doc/code changes to
land with Issue #13** (Tier 1: D1, D2, D3, D7, D9). Six **influence
the design of downstream Issues #14–#18** (D4, D5, D6, D8, D10, D11)
and are encoded here so those issues inherit the right starting state.

### Decisions landing with Issue #13

**D1. Embeddings → `voyage-finance-2`.** Replace `voyage-3` in
`design.md:201,210,219` and `ARCHITECTURE.md:185`. Used by
`extract/rag_retrieval.py`, `extract/entity_resolution.py`,
`agents/memo_retrieval.py`.

**D2. Reranker → `bge-reranker-v2-m3`.** Replace `bge-reranker-base` in
`design.md:212` and `ARCHITECTURE.md:101`. Used by
`extract/rag_retrieval.py` and `agents/memo_retrieval.py`.

**D3. Version pins.** `unstructured`, `voyageai`, and
`sentence-transformers` (which ships the reranker) all release weekly
with occasional breaking changes. Pin in `pyproject.toml` to the
versions Issue #13 lands against; update with intent, not via lockfile
drift.

**D7. `Chunk` carries metadata fields.** The dataclass produced by
`extract/chunking.py` becomes:

```python
@dataclass(frozen=True)
class Chunk:
    text: str
    section_name: str           # "Item 1A", "Item 7", "Item 8", "Item 7A", ...
    char_span: tuple[int, int]  # exact byte offsets into the raw HTML text
    token_count: int            # tiktoken cl100k_base, enforces ≤ 4K
    # New metadata fields (D7):
    ticker: str
    filing_date: date           # SEC accepted-date
    fiscal_period: str          # "FY2024", "Q3-2025", ...
    doc_type: str               # "10-K", "10-Q", "8-K", "S-1", "S-3"
    doc_id: str                 # content hash of raw source — joins to manifest
```

These fields are required, not optional. Tests assert all five
metadata fields are non-null on every Chunk.

**D9. char_span fidelity is a Tier 2 test, not a hope.** Issue #13's AC
already requires `source_text[chunk.char_span] == chunk.text`. Add to
that:

- A **post-validation** function `validate_char_spans(chunks, source_text)
  → list[Chunk]` that raises on mismatch — used both in the test and at
  runtime by `extract/chunking.py:parse_10k`. Failures route to
  `data/quarantine/{doc_id}.json` with the offending chunk + reason,
  consistent with the INV-2 quarantine pattern.
- A specific test for HTML edge cases that `unstructured` is known to
  normalize: `&nbsp;`, `&#8217;`, nested `<span>`s inside paragraphs,
  CDATA sections. The test fixture is constructed to include each.

### Decisions encoded here, applied by downstream issues

**D4. Parent-document retrieval (Issue #13 + #15).** `chunking.py`
returns **two** chunk levels:

- `ParentChunk` — section-respecting, ≤ 4K tokens, the context unit
  handed to extraction. Same metadata as above.
- `ChildChunk` — sentence-window subdivision (~400–600 tokens, never
  crosses parent boundary), the retrieval unit embedded into LanceDB.
  Carries `parent_id: str` referencing its parent's `doc_id +
  char_span` tuple.

Embedding (Issue #15) embeds `ChildChunk`s only. Retrieval (Issue #16)
matches at child level; returns parent-resolved hits to extraction.

**D5. Table-handling policy (Issues #13 + #19).** `extract/chunking.py`
**does not** RAG-embed `Table` elements from 10-K Item 8. Instead:

- `<table>...</table>` regions inside a section emit as standalone
  `ParentChunk`s (`section_name="Item 8"` for Item 8 tables) with the
  raw HTML attached as `table_html: str | None` (None on non-table
  chunks). INV-2 holds: `chunk.text == html[char_span] == table_html`.
- Structured financial extraction (Issue #19's 10-K worker) reads
  `table_html` directly via a typed Pydantic schema, bypassing RAG.
- The `MAX_PARENT_TOKENS` cap applies to **narrative parents only**.
  Tables can exceed the cap because (a) splitting raw HTML mid-table
  would produce invalid markup, and (b) the table path doesn't use
  the dense-retrieval token budget — extraction reads the raw HTML
  directly through a separate schema. Issue #13's tests assert the
  4K cap on narrative parents only.
- RAG retrieval ignores chunks where `table_html is not None`; if the
  extractor wants the underlying table, it dereferences `table_html`.

Net effect: tables stay in the chunk stream (preserving section
ordering and document coverage) but don't pollute the dense index.

**D6. Contextual-chunking cache key includes prompt version (Issue
#14).** The cache key for the contextual-chunking generation
(`extract/chunking_contextual.py`'s LLM call introduced by #14) is
`(parent_chunk_text, doc_metadata, contextual_prompt_version,
model_id)` — never just the chunk text. The `bump-prompt-version` skill
enforces the prompt-version side at edit time; the cache key enforces
the runtime side. INV-6 applies.

**D8. RRF weight tuning hook (Issue #16).** `extract/rag_retrieval.py`
exposes `bm25_weight: float = 1.0, dense_weight: float = 1.0` parameters
on the hybrid-retrieve function. Default is symmetric (the standard
RRF). Tuning is deferred to when there's eval data, but the surface is
present so we don't need a function-signature change later.

**D10. Single-shot cutoff is a tunable constant (Issue #19).** Define
`SINGLE_SHOT_TOKEN_CUTOFF = 100_000` in `extract/chunking.py` as a
module-level constant with a docstring stating it's calibrated, not
fundamental. Issue #19's 10-K worker reads it; Ragas/DeepEval evals
(Issue #20) include a single-vs-RAG comparison on docs near the
cutoff for retroactive calibration.

**D11. Cross-corpus index for Signal A1 (Issue #15 amendment).** In
addition to the per-doc LanceDB at `data/rag/{doc_id}.lance`, write
each `ChildChunk`'s embedding to a parallel per-corpus index at
`data/rag/_corpus_narrative.lance` (narrative sources: 10-K, 10-Q,
transcripts). Signal A1's retrieval (`signals/a1_supply_chain.py`
when implemented in W3) queries the corpus index directly with
`filing_date` + `doc_type` filters.

The per-corpus index is **write-only-on-extract**; no separate ingest
path. Issue #15's adapter writes both stores from the same
`embed(chunks)` call.

## Architecture

### `extract/chunking.py` (Issue #13 surface)

```python
# Public surface (after code review, 2026-05-25)
def parse_filing(*, html: str, metadata: ChunkMetadata) -> ChunkSet:
    """Parse SEC HTML into section-aware parent + child chunks.

    Raises ChunkValidationError if any chunk fails the char_span identity
    check. Callers route via `validate_or_quarantine_chunkset` rather
    than handling the exception manually.
    """

def validate_or_quarantine_chunkset(
    chunkset: ChunkSet,
    *,
    source_text: str,
    doc_id: str,
    quarantine_root: Path = DEFAULT_QUARANTINE_ROOT,
) -> ChunkSet | None:
    """Mirror of `extract.guardrails.validate_or_quarantine`. On INV-2
    failure: writes the quarantine record and returns None. Callers
    that get None MUST NOT persist any part of the chunkset downstream.
    """

def quarantine_chunkset(
    chunkset: ChunkSet, *, doc_id: str, source_text: str, reason: str,
    quarantine_root: Path = DEFAULT_QUARANTINE_ROOT,
) -> Path:
    """Write data/quarantine/chunking/<doc_id>.json. Requires explicit
    doc_id (no 'empty' fallback — caller always knows the document)."""

@dataclass(frozen=True)
class ChunkMetadata:
    ticker: str
    filing_date: date
    fiscal_period: str
    doc_type: str
    doc_id: str

@dataclass(frozen=True)
class ParentChunk:
    text: str
    section_name: str
    char_span: tuple[int, int]
    token_count: int
    table_html: str | None       # populated only for Item 8 tables (D5)
    metadata: ChunkMetadata

@dataclass(frozen=True)
class ChildChunk:
    text: str
    char_span: tuple[int, int]   # subset of parent's char_span
    token_count: int
    parent_id: str               # parent's (doc_id, char_span)
    section_name: str            # ADR D7 — copied from parent so LanceDB
                                 # can filter at index time
    from_table: bool             # ADR D5 — children of a table parent
                                 # are atomic (single child = parent)
    metadata: ChunkMetadata

@dataclass(frozen=True)
class ChunkSet:
    # tuples (not lists) so frozenness extends to the contents —
    # downstream consumers can't mutate parents/children in place.
    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]
```

`parse_filing` is pure (modulo a one-time spaCy warmup, lazy-loaded
inside the function): same `(html, metadata)` in → same `ChunkSet` out.
No network. No LLM calls. Tests pin `unstructured` to a specific
version (D3) so parser output is deterministic.

**Table-parent atomicity (D5 amendment, 2026-05-25).** Table parents
(`table_html is not None`) emit a single child equal to the parent.
Splitting at `</td>` seams produces fragments without closing
`</table>`, breaking D5's well-formed-HTML invariant. Children carry
`from_table=True` so Issue #16's retrieval can filter at the child
level without a JOIN.

**Cross-boundary tables (D5 amendment, 2026-05-25).** A `<table>` that
opens inside a section but `</table>` closes after is NOT emitted as a
table chunk. Clamping would produce malformed `table_html`. The open
`<table>` falls into the section's narrative; the close lands in the
next section's narrative.

**Routing wrapper (review finding, 2026-05-25).** Callers must use
`validate_or_quarantine_chunkset` rather than rolling their own
try/except around `parse_filing`. The wrapper mirrors
`guardrails.validate_or_quarantine`'s one-call contract so the INV-2
quarantine discipline is symmetric across both halves of the
invariant.

### Downstream surface (informational)

```
Issue #13 → ChunkSet              (parents + children, with metadata)
Issue #14 → ContextualChildChunk  (children with one-line context prepended)
Issue #15 → LanceDB(per-doc) + LanceDB(per-corpus narrative)  (D11)
Issue #16 → hybrid_retrieve(query, k, *, bm25_weight, dense_weight)  (D8)
Issue #17 → rerank with bge-reranker-v2-m3  (D2)
Issue #19 → routes < SINGLE_SHOT_TOKEN_CUTOFF to single-shot, else RAG  (D10)
```

## Consequences

### Positive

- RAG layer ships with a financial-domain stack on day 1 (D1, D2) at
  the same cost.
- Signal A1's cross-doc retrieval works without retroactive schema
  migration (D7, D11).
- Tables stop polluting the dense index; structured financials get a
  cleaner extraction path (D5).
- INV-2 contract is mechanically tested against `unstructured`'s known
  failure modes, not just a fixture-happy path (D9).

### Negative / accepted tradeoffs

- **`chunking.py` is materially more code than the issue body
  suggested** — two chunk types, table-html plumbing, char_span
  validation, metadata propagation. Estimated ~200-300 LOC for the
  module, plus ~150 LOC of tests. Within Issue #13's "medium" sizing
  but at the top of it.
- **Parent-document retrieval doubles LanceDB write volume.** Children
  are ~5–10× more numerous than parents (200-600 tokens vs ≤ 4K).
  Embedding cost rises proportionally (~$24 → ~$50 backfill); still
  comfortably inside §3's $250 envelope.
- **Per-corpus index (D11) is a write-time decision.** If we drop A1's
  cross-doc requirement later, the index becomes dead weight. Cost is
  a few MB on disk and one extra LanceDB write per chunk; reversible.
- **D5 means Issue #19's 10-K worker has two extraction paths**
  (narrative RAG + table structured). Documented here so the worker
  design doesn't get surprised in W2 D9.

### Out of scope for this ADR

- HyDE / multi-query expansion — extraction queries are well-formed
  Pydantic field descriptions; lift would be marginal.
- Deduplication of YoY-repeated boilerplate (risk factors copied
  across 10-Qs). Solvable later with a near-dup filter on `ChildChunk`
  text; not a v1 blocker.
- Hierarchical summarization for docs > 200K tokens. Only relevant if
  D10's cutoff calibration shifts; defer.

## Invariants touched

- **INV-2 (citation grounding).** D9 strengthens the char_span test
  with `unstructured`-specific edge cases and adds runtime validation
  with quarantine routing. The contract itself is unchanged.
- **INV-6 (determinism — version-pinned configs).** D6 names the
  contextual-chunking prompt version as part of the cache key, closing
  the silent-staleness path. D3 (version pins) reinforces parser
  determinism.

No other invariants touched.

## Rollback

Each decision is independently revertable; the ADR groups them but the
code does not entangle them.

| Decision | Rollback |
|---|---|
| D1 (`voyage-finance-2`) | Swap model string back to `voyage-3` in `extract/rag_retrieval.py:embed`. Re-embed affected corpus. |
| D2 (`bge-reranker-v2-m3`) | Swap model string back in `extract/rag_retrieval.py:rerank`. No re-embed needed. |
| D3 (pins) | Loosen `pyproject.toml` constraints. |
| D4 (parent-doc retrieval) | Single chunk level: emit only `ParentChunk`, drop `ChildChunk`. Schema migration on LanceDB. |
| D5 (table policy) | Re-emit tables as text chunks; route Item 8 through RAG. |
| D6 (cache key) | Remove prompt version from cache key. **Do not roll back** unless explicitly intended — it's an INV-6 defense. |
| D7 (metadata fields) | Drop the fields from `Chunk`. Migration cost as D4. |
| D8 (RRF weights) | Drop the params, hard-code to symmetric. |
| D9 (char_span tests) | **Do not roll back** — Tier 2 evidence for INV-2. |
| D10 (cutoff constant) | Inline `100_000` at use sites. |
| D11 (corpus index) | Stop writing the per-corpus store; remove the directory. |

## References

- Issue #13: `feat(extract): unstructured.io parsing + section-aware chunking`
- Issue #14: `feat(extract): contextual chunking (Anthropic pattern)`
- Issue #15: `feat(extract): LanceDB + Voyage embeddings adapter`
- Issue #16: `feat(extract): hybrid retrieval (BM25 + dense + RRF)`
- Issue #17: `feat(extract): BGE reranker on top-20 → top-5`
- Issue #18: `feat(extract): entity resolution`
- Issue #19: `feat(extract): 10-K, transcript, 8-K worker bodies`
- `docs/specs/2026-05-22-design.md` §8 (frozen; this ADR amends)
- Anthropic, *Introducing Contextual Retrieval* (2024) — the
  contextual-chunking pattern referenced in §8.1 (motivates D6)
