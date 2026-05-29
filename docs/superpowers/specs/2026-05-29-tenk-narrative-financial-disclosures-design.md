# 10-K narrative financial-disclosure signals (re-scope of issue #79)

**Status:** draft, 2026-05-29
**Supersedes:** the original framing of issue #79 ("complete TenKFinancials —
split into income/balance/cash-flow sub-schemas with classifier dispatch").
**Related:** `docs/specs/2026-05-28-xbrl-sql-layer-design.md` (the XBRL ingest
that obsoletes LLM extraction of structured financial line items).

## 1. Why this re-scope

Issue #79 as originally drafted proposed expanding the 10-K Item 8 LLM
table-extraction path from 10 fields to ~75 fields across three
sub-schemas, dispatched by a per-table classifier. The motivation was a
list of concrete signal needs (gross margin, free cash flow, accrual
quality, dilution tracking, leverage, liquidity, operating leverage,
non-cash adjustments).

One day before this re-scope, the XBRL SQL layer design landed
(`docs/specs/2026-05-28-xbrl-sql-layer-design.md`). It ingests SEC
CompanyFacts XBRL and exposes mart views (`income_statement_arq/mrq`,
`balance_sheet_arq/mrq`, `cash_flow_arq/mrq`) covering ~30–50 us-gaap
concepts that include every line item motivating issue #79: revenue,
COGS, gross profit, R&D, SG&A, opex, operating income, net income, EPS,
shares, cash, receivables, inventory, current assets, PP&E, goodwill,
intangibles, total assets, AP, short/long-term debt, current/total
liabilities, equity, CFO, capex, free cash flow, dividends paid.

The XBRL path is authoritative (filer-tagged us-gaap), deterministic,
free, and handles restatements via AR/MR view variants. The current
10-field LLM path is probabilistic, costs ~$0.05/10-K, and produces the
same numbers for the same filings (universe_v1 is 80 post-2009
us-gaap-era filers — 100% within XBRL coverage; the XBRL spec's NG1
explicitly scopes itself to this set).

The right move is therefore not to expand the LLM extraction but to
narrow it: delete the redundant table-extraction path, and use the
saved budget on the narrow band of signals XBRL definitionally cannot
give — auditor / management narrative disclosures.

## 2. Scope

### In scope

1. **Delete the LLM Item 8 table-extraction path.** Removed types,
   prompts, worker functions, routing rows, and tests are enumerated in
   §5.
2. **Add three narrative-only signals** to `TenKOutput`. Populated by
   both the single-shot narrative path (`TEN_K_NARRATIVE_PROMPT`) and
   the per-field RAG path (`_extract_ten_k_rag` in
   `src/auto_research/extract/workers/ten_k.py`), matching the
   existing convention for narrative fields like `guidance_tone` /
   `accrual_flags`.
   - `going_concern: Claim | None`
   - `icfr_material_weaknesses: list[Claim]`
   - `critical_accounting_estimate_changes: list[Claim]`
3. Three new RAG partial models, three new entries in
   `TEN_K_NARRATIVE_FIELD_CONFIGS`, three new routing-table rows for
   the RAG calls, single-shot prompt update, matching unit tests on
   both paths.

### Out of scope (deferred)

- The 75-field structured sub-schema expansion (`TenKIncomeStatement`,
  `TenKBalanceSheet`, `TenKCashFlowStatement`) — superseded by XBRL.
- The per-table `_classify_table` classifier and dispatch architecture
  — only valuable if we were still LLM-extracting structured tables.
- Non-GAAP storytelling intensity — soft signal; defer until the eval
  suite (#20) can score it.
- Restatement narrative — pair with XBRL MR-vs-AR diff; needs XBRL
  ingest live first.
- Segment narrative — wait for XBRL v1.1 segment axis (the XBRL spec
  itself defers segments to v1.1).
- Covenant headroom — overlaps `risk_factor_deltas`; defer.

## 3. New field semantics

| Field | Source sections in a 10-K | `field_description` (drives the prompt's `{field_description}` slot) | `retrieval_query` (drives rerank in `_extract_ten_k_rag`) |
|---|---|---|---|
| `going_concern` | Item 8 audit report; Item 7 liquidity discussion | A single Claim quoting the auditor's "substantial doubt" sentence verbatim, or `None` when the audit report carries an unqualified opinion. Do not paraphrase — quote the actual disclaimer sentence. | "Does the auditor's report or liquidity discussion express substantial doubt about the company's ability to continue as a going concern?" |
| `icfr_material_weaknesses` | Item 9A management's ICFR assessment | A list of Claims, one per distinct material weakness disclosed in management's Item 9A internal-controls report. Empty list when management concludes ICFR is effective with no material weaknesses identified. | "Does management's Item 9A internal-controls report identify any material weaknesses in ICFR?" |
| `critical_accounting_estimate_changes` | Item 7 MD&A "Critical Accounting Estimates"; Item 8 Significant Accounting Policies notes | A list of Claims for estimates flagged as requiring significant judgment AND where management indicates a change vs the prior year (new estimate, methodology change, materially different assumptions). Empty list when no YoY change is flagged. | "Which critical accounting estimates did management flag as new, changed, or requiring different assumptions versus the prior year?" |

All three fields' `Claim.confidence` is the categorical `Literal["high",
"medium", "low"]` already required by the shared narrative prompt — no
new schema surface. Downstream signal code can threshold on `high` if
false-positive rate becomes an issue.

## 4. Architecture (no new architecture)

`TenKOutput` is produced by **two** narrative paths today — the
single-shot path (`TEN_K_NARRATIVE_PROMPT` against the full raw doc,
target schema `TenKOutput`) and the RAG per-field path
(`TEN_K_NARRATIVE_FIELD_PROMPT` looped over
`TEN_K_NARRATIVE_FIELD_CONFIGS`, target schema per partial). Every
narrative field on `TenKOutput` is populated by both paths today
(`guidance_tone`, `accrual_flags`, `supplier_mentions`,
`customer_mentions`, `risk_factor_deltas`) — the three new fields
follow the same convention.

The new fields plug into both paths unchanged in shape:

- **RAG path.** `TEN_K_NARRATIVE_FIELD_CONFIGS`
  (`prompts/ten_k_narrative_field.py`) gains three trailing entries.
  The loop in `_extract_ten_k_rag` iterates them like the existing five
  — one Anthropic call per field, scoped to the top reranked parents
  returned by `retrieve_fn(query)`, staged cache writes committed only
  after all fields succeed AND the cross-partial identity check passes.
- **Single-shot path.** `TEN_K_NARRATIVE_PROMPT`
  (`prompts/ten_k_narrative.py`) is updated to extract the three new
  fields. The "do not populate `financials`" instruction is removed
  (the field no longer exists on `TenKOutput`).
- Each new RAG partial model carries the identity fields (`cik`,
  `accession_number`, `fiscal_period_end`) plus exactly one narrative
  field — same shape as the existing `TenK*Partial` set.
- Routing-table rows go in `_models.py` under the existing 10-K block;
  all three route to `_LOCAL_QWEN_35B_MOE` — the smoke-tested locked
  local stack (cost-model doc §10.5). Each field is a high-volume
  templated-pattern call with bounded output shape (one Claim or a
  short list of Claims), which is the workload the local stack was
  validated for. The routing rows apply to the RAG per-field calls;
  the single-shot call already routes via
  `_NARRATIVE_DEFAULT_TASK = "supplier_mentions"` and that is unchanged.
  The local routing is gated by `_ALLOWED_LOCAL_ROWS` in
  `tests/unit/test_extract_local_dispatch.py` — adding a fourth local
  row in a future change requires extending that allowlist with
  smoke-test evidence.
- INV-2 grounding inherits from `Claim`/`Citation` — `_resolve_spans`
  in `_common.py` handles whitespace-flexible regex resolution; ZERO
  / AMBIGUOUS match handling unchanged.
- Per-field quarantine semantics inherited — one field's failure
  quarantines that field's call only and aborts the whole 10-K
  (consistent with the existing loop's behavior so partial output
  never lands).

## 5. Deletion checklist (line-level callouts)

- `src/auto_research/extract/schemas.py`
  - Drop class `FinancialLineItem` (currently at `schemas.py:179`).
  - Drop class `TenKFinancials` (currently at `schemas.py:196`).
  - Drop field `TenKOutput.financials` (currently at `schemas.py:254`)
    and its surrounding comment block.
  - Drop `"FinancialLineItem"` and `"TenKFinancials"` from `__all__`.
- `src/auto_research/extract/prompts/ten_k_financials.py` — delete the
  file. Remove the re-export from `prompts/__init__.py` if present
  (verify at implementation time).
- `src/auto_research/extract/workers/ten_k.py`
  - Drop the `TEN_K_FINANCIALS_PROMPT*` imports (`ten_k.py:48–50`).
  - Drop `TenKFinancials` from the schemas import (`ten_k.py:61`).
  - Drop module constants `_FINANCIALS_TASK`, `_FINANCIALS_MAX_TOKENS`.
  - Delete `_render_table_html_to_text`, `_extract_item8_financials`,
    `_merge_financials`.
  - Delete the entire "Item 8 financials" branch at the end of
    `extract_ten_k` (`ten_k.py:348–377`) — function returns `narrative`
    directly after the narrative branch resolves.
  - Drop the lazy `from bs4 import BeautifulSoup` import inside the
    deleted `_render_table_html_to_text`. Verify no other call site in
    `extract/` needs `bs4` after the deletion (grep at implementation).
- `src/auto_research/_models.py`
  - Delete row `("ten_k", "financials"): _HAIKU` (currently at
    `_models.py:63`) and the two-line comment above it.
- `src/auto_research/extract/prompts/ten_k_narrative.py`
  - Remove the "do not populate `financials`" instruction from
    `TEN_K_NARRATIVE_PROMPT` (the field no longer exists on
    `TenKOutput`).
- `tests/unit/test_extract_prompts.py`
  - Drop the `from auto_research.extract.prompts.ten_k_financials import …`
    import.
  - Drop `test_ten_k_financials_prompt_exports` and any helper used only
    by it.
  - Drop the `"financials" in TEN_K_NARRATIVE_PROMPT` assertion at
    `test_extract_prompts.py:113` — the narrative prompt no longer
    mentions `financials` after deletion.
- `tests/unit/test_models.py`
  - Drop `test_route_model_routes_ten_k_financials_to_haiku`.
- `tests/unit/test_extract_worker_ten_k.py`
  - Drop the `TenKFinancials` import and the `_merge_financials` import.
  - Drop the `_valid_financials` test helper.
  - Drop `test_ten_k_item8_financials_extracted_from_table_html` and any
    other test asserting `out.financials is None` / `is not None`.
- Pre-deletion sweep
  - Run `rg "TenKFinancials|FinancialLineItem|TEN_K_FINANCIALS|_extract_item8_financials|_merge_financials|_render_table_html_to_text"`
    across `src/`, `tests/`, `eval/`, `scripts/`, `docs/`. Any hit
    outside the deletion set is a consumer the issue's framing missed —
    surface back to the user before continuing the implementation.

## 6. Additions checklist

- `src/auto_research/extract/schemas.py`
  - Add three partial models: `TenKGoingConcernPartial`,
    `TenKIcfrMaterialWeaknessesPartial`,
    `TenKCriticalAccountingEstimateChangesPartial`. Each follows the
    existing `TenK*Partial` shape — `model_config = _FROZEN_STRICT`,
    `SCHEMA_VERSION: ClassVar[str] = "v1"`, the three identity fields,
    and the single narrative field.
  - Add three fields to `TenKOutput`: `going_concern: Claim | None`,
    `icfr_material_weaknesses: list[Claim]`,
    `critical_accounting_estimate_changes: list[Claim]`. No defaults
    (consistent with the existing narrative fields like
    `guidance_tone` / `accrual_flags`) — both narrative paths
    (single-shot and RAG) are required to populate them on every
    successful extraction.
  - Add the three partial class names to `__all__`.
- `src/auto_research/extract/prompts/ten_k_narrative.py`
  - Add per-field extraction instructions for the three new fields to
    `TEN_K_NARRATIVE_PROMPT` (single-shot path). Mirror the per-field
    semantics in §3 so the single-shot and RAG paths produce
    interchangeable output. Keep the new instructions at the end of
    the prompt so the cached prefix on the single-shot call is not
    invalidated structurally.
- `src/auto_research/extract/prompts/ten_k_narrative_field.py`
  - Import the three new partials.
  - Append three `TenKNarrativeFieldConfig` entries to
    `TEN_K_NARRATIVE_FIELD_CONFIGS`. New entries go at the end (the
    existing comment marks the tuple order as load-bearing for the
    cache namespace; appending preserves the cache state of the
    existing five fields).
- `src/auto_research/_models.py`
  - Add three routing rows in the 10-K block, all `_HAIKU`. Comment
    cites the cost-model doc §10.5 "Locked stack" — the smoke-tested
    Qwen 35B-MoE backend that these three high-volume templated-pattern
    fields route to.

## 7. Tests

- `tests/unit/test_models.py` — three assertions:
  `route_model("ten_k", "going_concern") == _HAIKU`, same for
  `icfr_material_weaknesses` and `critical_accounting_estimate_changes`.
- `tests/unit/test_extract_prompts.py`
  - Assert the three new entries exist in
    `TEN_K_NARRATIVE_FIELD_CONFIGS` and the count matches.
  - Assert each new entry's `retrieval_query` is non-empty and contains
    a section reference (Item 8 / Item 9A / Item 7) so retrieval drift
    is mechanically caught.
  - Assert each `field_description` references the source section
    (Item 8 audit / Item 9A / Item 7 MD&A or footnotes).
- `tests/unit/test_extract_worker_ten_k.py`
  - One per-field happy-path RAG test for each of the three new fields:
    `retrieve_fn` stubbed to return parents with planted language; the
    mocked extraction client returns a valid partial; assert the merged
    `TenKOutput` carries the right `Claim` shape.
  - "Going-concern absent" test: retrieved passage carries an
    unqualified audit opinion; partial returns `going_concern=None`;
    merged output carries `None`.
  - "ICFR effective" test: retrieved Item 9A passage indicates ICFR is
    effective with no material weaknesses; partial returns
    `icfr_material_weaknesses=[]`; merged output carries `[]`.
  - "Critical estimates unchanged" test: Item 7 passage discusses
    critical estimates but flags no YoY change; partial returns
    `critical_accounting_estimate_changes=[]`; merged output carries
    `[]`.
  - **Single-shot path coverage:** extend the existing single-shot
    narrative test (`_valid_narrative` fixture) to include the three
    new fields in the planted LLM response, and assert the resulting
    `TenKOutput` carries them. A separate single-shot test should
    populate `going_concern` with a non-None Claim to exercise the
    single-shot positive path for the new fields too.
  - Update the existing identity-agreement test to include the new
    fields in the count of partials that must agree on `(cik,
    accession_number, fiscal_period_end)`.

## 8. INV compliance

- **INV-1 (PIT discipline).** Not in scope. `TenKOutput` does not write
  to Feast directly; PIT is enforced at the FeatureView boundary.
- **INV-2 (citation grounding).** Inherited via `Claim`/`Citation`
  shape. `_resolve_spans` handles regex resolution; AMBIGUOUS / ZERO
  match semantics unchanged.
- **INV-6 (version pinning).** New partials declare
  `SCHEMA_VERSION = "v1"` (initial). Per the project's pre-deployment
  prompt-version policy, no bump on `TenKOutput.SCHEMA_VERSION`,
  `TEN_K_NARRATIVE_FIELD_PROMPT_VERSION`, or
  `TEN_K_NARRATIVE_PROMPT_VERSION` — both narrative prompt files
  change byte content (removed `financials` instruction; added 3 new
  fields' instructions on the single-shot side; 3 new
  `TenKNarrativeFieldConfig` entries on the RAG side), but nothing
  downstream consumes the contract yet, so version-bumping would
  invalidate cache entries the project never paid for.

## 9. Cost

Per long 10-K (RAG path, post-#78 PR-B economics ~$0.47/filing):

| Change | Per-filing delta |
|---|---|
| Delete Item 8 LLM table call | −~$0.05 |
| Add 3 local Qwen 35B-MoE field calls (`going_concern`, `icfr_material_weaknesses`, `critical_accounting_estimate_changes`) | \$0 incremental Anthropic spend (local stack) |
| **Net** | **−~$0.05** |

The local-stack calls have no per-call Anthropic budget; the only
incremental cost is on-prem GPU time on the locked Mac M2 / vllm-mlx
host (cost-model doc §10.5).

At 500 long 10-Ks per backfill cycle: ~$25 saved per cycle. Short
10-Ks (single-shot path) see the same direction.

## 10. Sequencing

- Independent of `docs/specs/2026-05-28-xbrl-sql-layer-design.md`
  ingest implementation — no shared modules. Either can land first.
- Dependencies on #78 PR-A (tool_use) and #78 PR-B (Option A per-field
  RAG) are already satisfied — both merged on the current branch.
- Should land before the eval suite (#20) so DeepEval baselines bake in
  the final `TenKOutput` shape (no `financials`, three new narrative
  fields).
- Should land before backfill (#22) so the first backfill writes the
  final shape.
- Closes #79.

## 11. Risks

- **Going-concern is rare in `universe_v1`.** Modal outcome is
  `going_concern=None` because the universe is large-cap mature
  filers. The signal value is in the rare positive case. At least one
  test fixture must include planted "substantial doubt" language so
  the extractor's positive path is exercised.
- **Critical-estimate-change calls are subjective.** Mitigated by the
  inherited categorical `confidence` field — downstream signal code
  can threshold on `high` if false positives surface.
- **Item 9A retrieval coverage.** The backfill orchestrator's retrieve
  function (out of this PR's scope) must surface Item 9A passages in
  the rerank top-k. Verify during the eval-suite work (#20); add a
  retrieval-coverage probe if Item 9A is consistently absent.

## 12. Acceptance criteria

- `make check` green (pyright strict, pytest, routing-table coverage
  test, prompt-shape tests).
- `rg "TenKFinancials|FinancialLineItem|TEN_K_FINANCIALS|_extract_item8_financials|_merge_financials"`
  inside `src/` and `tests/` returns zero hits after the PR.
- `TenKOutput` carries the three new fields with the expected types.
- Each new field has a passing happy-path RAG test and a passing
  "absent" test (returns `None` / `[]`).
- Routing-table coverage test asserts all three new `(ten_k, …)` rows.
- Issue #79 closes referencing this spec.
