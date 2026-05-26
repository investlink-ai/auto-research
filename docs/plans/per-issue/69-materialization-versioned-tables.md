# Issue 69 — Materialization-versioned LanceDB tables + active-pointer config

**Tier:** 1. `extract/embeddings.py` is not on the §3 sensitive-path list (the
guarded extract paths are `guardrails.py`, `schemas.py`, `chunking.py`); this
change is operational layout, not contract-shape.

**Branch / worktree:** `feat/69-materialization-versioned-tables` at
`~/Documents/projects/auto-research/.worktree/69-materialization-versioned-tables/`.

This per-issue plan is disposable; it survives only until PR merge.

---

## 1. Design decisions settled

| Question | Decision |
|---|---|
| Version-token format | `compute_materialization_version(chunker_v, contextual_prompt_v, embed_model_v) → first 8 hex chars of `sha256(\|)`. Pure function; deterministic; collision risk acceptable at the dozen-or-so versions we will ever produce. |
| Table names | Per-doc: `f"{doc_id}__{materialization_version}"`. Per-corpus: `f"_corpus_narrative__{materialization_version}"`. Separator `__` chosen to be distinguishable from any character that can appear in a doc_id (EDGAR accession numbers and tickers don't contain double-underscore). |
| Active pointer location | `data/rag/active_materialization.json` (singular). |
| Active pointer payload | `{"version": <hash>, "embed_model_version": <full token>, "promoted_at": <ISO-8601>, "manifest_count": <int>}`. `embed_model_version` is the full `embed_model_version()` token (`"voyage:voyage-finance-2:v1"`) so the read-path mismatch guard is unambiguous. |
| Promotion history | Separate file `data/rag/promotion_history.json`, append-only JSON list of `ActiveMaterialization` records. Read by `gc-materialization` for chronological "keep last N" semantics. |
| Build path (embed/reembed) | Writes to `{name}__{self.materialization_version}` ALWAYS, regardless of which version is active. Per issue: "Build path ignores it (writes to its own version namespace)." |
| `reembed_doc()` source | Reads source rows from `{doc_id}__{active.version}` (active pointer required to exist), writes destination to `{doc_id}__{self.materialization_version}`. Raises if `active.version == self.materialization_version` (nothing to do — the build path already produces this materialization). |
| Read path (query / bm25_query) | If `active_materialization.json` exists: open `{name}__{active.version}`; assert `active.embed_model_version == self.embed_model_version` and raise loudly on mismatch (feedback-embedding-vector-space-consistency). If pointer missing: fall back to `{name}__{self.materialization_version}` (lets fresh installs + tests work without a manual promote step). |
| OTel | Add `embedding.materialization_version` to `extract.embed`, `extract.reembed`, `extract.embed_query`, `extract.bm25_query` spans. |
| Migration | Not needed. Project is in pre-data development scope; any legacy `{name}.lance` tables are rebuilt by re-running the embed sweep under the versioned layout. |
| GC ordering | Promotion-history-ordered ("last N versions promoted"), NOT lexicographic on the hash (hashes are random). Active version is always preserved regardless of where it sits in history. |

### Out of scope (matches the issue body)

- Dual-write canary at query time.
- Hot/cold recency-partitioned materializations.
- Cross-cloud replication.
- Schema-shape changes to the `text`/`vector`/metadata columns.

---

## 2. Files touched

```
src/auto_research/extract/materialization.py          (new)
src/auto_research/extract/embeddings.py               (table-naming + read-path routing)
src/auto_research/cli.py                              (3 new subcommands)
docs/decisions/2026-05-24-rag-enhancements.md         (D11 amendment + rollback row)
tests/unit/test_materialization.py                    (new)
tests/unit/test_embeddings.py                         (additions, ~6 new tests)
tests/unit/test_cli.py                                (additions, ~5 new tests)
tests/integration/test_embeddings_vcr.py              (1 new end-to-end test)
```

Module diagram (delta, not absolute):

```
                                            ┌────────────────────────┐
embed() / reembed_*()  ──writes──>          │  {name}__{m_version}   │
                                            │     .lance tables      │
query() / bm25_query() ──reads───>          └────────────────────────┘
                                                       ▲
                                                       │ table name
                                                       │ resolution
                                            ┌──────────┴─────────────┐
                                            │  materialization.py    │
                                            │  • compute_…_version() │
                                            │  • read_active(…)      │
                                            │  • atomic write_active │
                                            │  • promotion_history   │
                                            │  • list_materializations
                                            └────────────────────────┘
```

---

## 3. Step plan (TDD)

Each step is `Step → verify` per AGENTS.md §4 working style.

1. **materialization.py pure helpers.** Write `compute_materialization_version`, `ActiveMaterialization`, atomic read/write, history append. → `tests/unit/test_materialization.py` covers deterministic hashing, atomic-flip mid-failure (kill the tmp file → original survives), history append + read.

2. **Versioned table naming in `embed()`.** Route writes through `materialization_version`. → `tests/unit/test_embeddings.py::test_embed_writes_to_versioned_table` opens `{doc_id}__{version}.lance` directly; old `{doc_id}.lance` no longer exists.

3. **Read-path resolution.** `query()` and `bm25_query()` look up active pointer, fall back to own version, raise on embed-model-version mismatch. → unit tests for all three cases.

4. **Reembed source/dest split.** `reembed_doc()` reads `{doc_id}__{active.version}` and writes `{doc_id}__{self.materialization_version}`; raises if no active or if active matches own. → 3 unit tests.

5. **OTel attribute on all four spans.** → assert via the in-memory span exporter pattern already in `tests/unit/test_telemetry.py` (or via mock if simpler).

6. **CLI: `list-materializations`.** → CLI test: build two materializations in tmp, list returns both with row counts and active flag.

7. **CLI: `promote-materialization`.** Validates completeness against manifest, validates dim, atomic flip. → tests: happy path; partial namespace refused; dim mismatch refused; flip-mid-failure preserves previous pointer.

8. **CLI: `gc-materialization`.** Drops old non-active versions beyond `--keep-last N`. → test: 3 historical versions + 1 active, `--keep-last 2` drops the oldest, keeps active + 1 previous.

9. **Migration script.** → unit test (idempotent: run twice, second is no-op).

10. **ADR amendment.** Update D11 with the new layout; add `Rollback` row.

11. **Integration test:** end-to-end embed → promote → query loop against tmp LanceDB using BGE backend (hermetic, no network).

12. **`make quick` + targeted pytest.** Green required before commit.

---

## 4. PR body skeleton

```
Closes #69.

## Summary
- Per-doc and per-corpus LanceDB tables are now versioned by a
  `materialization_version` token (sha256-8 of chunker + contextual-prompt +
  embed-model version triple).
- New `auto-research extract` subcommands: `list-materializations`,
  `promote-materialization`, `gc-materialization`.
- Migration script removed from scope: pre-data development, legacy
  tables are rebuilt by re-running the embed sweep.
- ADR D11 amended; rollback documented.

## AC mapping
- Table naming scheme — `extract/embeddings.py` `_versioned_*_table_name`
- Active-pointer schema doc — `extract/materialization.py` `ActiveMaterialization` docstring
- 3 CLI subcommands — `cli.py::extract_{list,promote,gc}_materialization`
- Promote completeness/dim validation — `cli.py::extract_promote_materialization` + `tests/unit/test_cli.py::test_promote_materialization_*`
- Atomic flip — `extract/materialization.py::write_active_materialization` + `tests/unit/test_materialization.py::test_atomic_flip_*`
- Migration script — N/A (dropped from scope; rebuild rather than migrate)
- "embed-to-inactive doesn't perturb active query" — `tests/unit/test_embeddings.py::test_embed_to_inactive_namespace_does_not_perturb_active_query`
- "full embed → promote → query loop" — `tests/integration/test_embeddings_vcr.py::test_embed_promote_query_loop`
- ADR amendment — `docs/decisions/2026-05-24-rag-enhancements.md` D11 section
- OTel attribute — search for `embedding.materialization_version` in the diff

## Verification
- `make quick` clean.
- `pytest tests/unit/test_materialization.py tests/unit/test_embeddings.py tests/unit/test_cli.py` green.
- Integration: `pytest tests/integration/test_embeddings_vcr.py::test_embed_promote_query_loop` green.
```
