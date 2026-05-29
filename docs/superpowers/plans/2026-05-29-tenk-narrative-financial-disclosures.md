# 10-K Narrative Financial-Disclosure Signals (re-scope of #79) — Implementation Plan

> **Historical note (post-merge):** the routing rows for the three new
> fields in Tasks 3 / 4 / 5 below were originally added as `_HAIKU` and
> subsequently flipped to `_LOCAL_QWEN_35B_MOE` (the locked Mac M2 /
> vllm-mlx Qwen 35B-MoE stack) on top of #84's provider-agnostic
> dispatch. The unit-test assertions, `_ALLOWED_LOCAL_ROWS` gate, and
> the live smoke under `tests/live/test_ten_k_local_qwen_smoke.py`
> reflect the final state; the canonical design is in the spec, not
> this plan. Read the plan body for the TDD step sequence and ignore
> the `_HAIKU` literal in Tasks 3 / 4 / 5 — the symbol is now
> `_LOCAL_QWEN_35B_MOE`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the redundant LLM Item 8 table-extraction path on the
10-K worker (XBRL ingest now covers it), and add three narrative-only
`Claim`-bearing fields to `TenKOutput` that XBRL cannot give —
`going_concern`, `icfr_material_weaknesses`,
`critical_accounting_estimate_changes`.

**Architecture:** No new architecture. The three new fields plug into
the existing per-field RAG loop in `_extract_ten_k_rag` via three new
`TenKNarrativeFieldConfig` entries, three new `TenK*Partial` schemas,
and three new `_models.py` routing rows. The single-shot narrative
path's `TEN_K_NARRATIVE_PROMPT` is updated to extract the same three
fields. Per the project's pre-deployment prompt-version policy, no
`_VERSION` constants are bumped.

**Tech Stack:** Python 3, Pydantic v2 (frozen, `extra='forbid'`), Anthropic
SDK with `tool_use`, pytest with `pytest-mock`, `uv` for dependency
management, `make check` (ruff + mypy + pytest) as the merge gate.

**Spec:**
`docs/superpowers/specs/2026-05-29-tenk-narrative-financial-disclosures-design.md`.
Read it once before starting; it has the rationale and the design-level
detail. This plan is the mechanical execution checklist.

---

## File Structure

**Files to delete entirely**

- `src/auto_research/extract/prompts/ten_k_financials.py`

**Files to modify (deletions)**

- `src/auto_research/extract/schemas.py` — remove `FinancialLineItem`,
  `TenKFinancials`, `TenKOutput.financials`, and matching `__all__`
  entries.
- `src/auto_research/extract/workers/ten_k.py` — remove the Item 8
  branch, the prompt+schema imports, the rendering+merging+extraction
  helpers, and the financials task constants.
- `src/auto_research/_models.py` — remove the
  `("ten_k", "financials")` routing row.
- `src/auto_research/extract/prompts/ten_k_narrative.py` — remove the
  "DO NOT populate `financials`" instruction (the field will not
  exist).
- `tests/unit/test_extract_schemas.py` — remove the
  `TenKFinancials`/`FinancialLineItem` test block + imports.
- `tests/unit/test_extract_prompts.py` — remove the
  `ten_k_financials` import + `test_ten_k_financials_prompt_exports`
  + the `"financials" in TEN_K_NARRATIVE_PROMPT` assertion.
- `tests/unit/test_models.py` — remove
  `test_route_model_routes_ten_k_financials_to_haiku`.
- `tests/unit/test_extract_worker_ten_k.py` — remove the
  `TenKFinancials`/`FinancialLineItem`/`_merge_financials`/`_render_table_html_to_text`
  imports + `_valid_financials` helper +
  `test_ten_k_item8_financials_extracted_from_table_html` +
  `test_render_table_html_strips_tags_and_bridges_cells` +
  `test_merge_financials_takes_first_non_none_per_field` + every
  `assert out.financials is None / is not None` line in surviving tests.

**Files to modify (additions)**

- `src/auto_research/extract/schemas.py` — add three partial models
  (`TenKGoingConcernPartial`, `TenKIcfrMaterialWeaknessesPartial`,
  `TenKCriticalAccountingEstimateChangesPartial`), add three required
  fields to `TenKOutput`, extend `__all__`.
- `src/auto_research/_models.py` — add three Haiku routing rows.
- `src/auto_research/extract/prompts/ten_k_narrative_field.py` — add
  three `TenKNarrativeFieldConfig` entries (RAG path).
- `src/auto_research/extract/prompts/ten_k_narrative.py` — add
  per-field extraction instructions for the three new fields
  (single-shot path).
- `tests/unit/test_models.py` — three routing assertions.
- `tests/unit/test_extract_schemas.py` — three partial-schema
  construction tests + updated `TenKOutput` construction in surviving
  tests.
- `tests/unit/test_extract_prompts.py` — assertions that
  `TEN_K_NARRATIVE_FIELD_CONFIGS` carries the three new entries with
  retrieval-query section references; assertions that
  `TEN_K_NARRATIVE_PROMPT` names the three new fields.
- `tests/unit/test_extract_worker_ten_k.py` — three RAG-path
  happy-path tests + three "absent" tests + single-shot coverage
  extensions.

---

## Task 1: Create worktree and feature branch

**Files:**
- New worktree at the path the project canonical layout uses for #79.

- [ ] **Step 1: Confirm working directory is clean**

Run:
```
git status --short
```
Expected: only `learning/` working-tree changes (untracked files) and
`.claude/scheduled_tasks.lock` — no staged or modified files in `src/`
or `tests/`. If `src/` or `tests/` are dirty, surface to user before
continuing.

- [ ] **Step 2: Invoke the `worktree` skill**

Use the `worktree` skill with issue number `79` and the description
"re-scope #79 — drop LLM Item 8 extraction, add 3 narrative-only
signals". The skill will create a per-issue worktree at the
project-canonical path with the right branch name. Skill maps to spec
§23.4. Do NOT create the worktree manually with `git worktree add`;
let the skill do it so the branch-name + path conventions match.

- [ ] **Step 3: Confirm starting commit**

In the new worktree, run:
```
git log --oneline -1
```
Expected: the commit at the tip of `main` (the spec commit
`a9eccc7 docs(specs): re-scope #79 …` or its descendant).

---

## Task 2: Delete the LLM Item 8 table-extraction path

**Goal:** Remove every line of code/test/prompt referenced in the spec
§5 deletion checklist plus `tests/unit/test_extract_schemas.py` (which
the spec missed; sweep surfaced it). After this task, `make check` is
green and no source file references the deleted symbols. This is a
multi-file edit landing in **one commit**.

**Files:**
- Delete: `src/auto_research/extract/prompts/ten_k_financials.py`
- Modify: `src/auto_research/extract/schemas.py`
- Modify: `src/auto_research/extract/workers/ten_k.py`
- Modify: `src/auto_research/extract/prompts/ten_k_narrative.py`
- Modify: `src/auto_research/_models.py`
- Modify: `tests/unit/test_extract_schemas.py`
- Modify: `tests/unit/test_extract_prompts.py`
- Modify: `tests/unit/test_extract_worker_ten_k.py`
- Modify: `tests/unit/test_models.py`

- [ ] **Step 1: Run the pre-deletion sweep**

```
rg -n "TenKFinancials|FinancialLineItem|TEN_K_FINANCIALS|_extract_item8_financials|_merge_financials|_render_table_html_to_text" -tpy src/ tests/ eval/ scripts/
```

Expected hits (and only these) — anywhere else is a missed consumer
that requires surfacing to the user:
- `src/auto_research/extract/schemas.py` (class defs + `__all__` entries
  + the `TenKOutput.financials` field).
- `src/auto_research/extract/workers/ten_k.py` (imports + helpers + the
  `extract_ten_k` Item 8 branch).
- `src/auto_research/extract/prompts/ten_k_financials.py` (the whole
  module).
- `src/auto_research/_models.py` (one routing row).
- `tests/unit/test_extract_schemas.py` (lines 26, 31, 429–599 region).
- `tests/unit/test_extract_prompts.py` (lines 19–22 import, 113
  assertion, 117–130 test).
- `tests/unit/test_extract_worker_ten_k.py` (lines 30–35 imports, 77
  helper, 168 / 188 / 338 incidental asserts, 497 Item 8 test, 527
  rendering test, 544 merge test).
- `tests/unit/test_models.py` (line 50 test).

`docs/plans/per-issue/*` hits are historical implementation plans for
already-merged work — leave them alone.

- [ ] **Step 2: Delete `src/auto_research/extract/prompts/ten_k_financials.py`**

```
rm src/auto_research/extract/prompts/ten_k_financials.py
```

Verify the package `__init__.py` does NOT re-export it (it does not at
audit time; if a re-export got added, drop it):

```
grep -n "ten_k_financials" src/auto_research/extract/prompts/__init__.py
```
Expected: no output.

- [ ] **Step 3: Modify `src/auto_research/_models.py` — drop the routing row**

Delete the three lines at `_models.py:61–63`:

```python
    # Item 8 financials: templated table → JSON; high-volume pattern
    # recognition per §7.3 ⇒ Haiku.
    ("ten_k", "financials"): _HAIKU,
```

- [ ] **Step 4: Modify `src/auto_research/extract/workers/ten_k.py` — drop imports**

Replace this import block (currently at `ten_k.py:48–50`):

```python
from auto_research.extract.prompts.ten_k_financials import (
    TEN_K_FINANCIALS_PROMPT,
    TEN_K_FINANCIALS_PROMPT_VERSION,
)
```

with nothing (remove the block).

Then change the `from auto_research.extract.schemas import …` line at
`ten_k.py:61` from:

```python
from auto_research.extract.schemas import TenKFinancials, TenKOutput
```

to:

```python
from auto_research.extract.schemas import TenKOutput
```

- [ ] **Step 5: Modify `src/auto_research/extract/workers/ten_k.py` — drop constants and helpers**

Remove these two module constants (currently near `ten_k.py:73–75`):

```python
_FINANCIALS_TASK = "financials"
_FINANCIALS_MAX_TOKENS = 4096
```

Remove the entire `_render_table_html_to_text` function
(`ten_k.py:92–101`), the entire `_extract_item8_financials` function
(`ten_k.py:104–136`), and the entire `_merge_financials` function
(`ten_k.py:139–160`).

- [ ] **Step 6: Modify `src/auto_research/extract/workers/ten_k.py` — drop the Item 8 branch in `extract_ten_k`**

In `extract_ten_k`, find the block currently at `ten_k.py:347–377` —
the section starting with the comment

```python
    # 2. Item 8 financials: independent of narrative path. Iterates EVERY
```

and ending with

```python
    return narrative.model_copy(update={"financials": financials})
```

Delete the entire block. The function's tail must end with the
narrative return path. Specifically, the function's existing line at
`ten_k.py:345–346`:

```python
    if narrative is None:
        return None
```

becomes the last meaningful statement before the function returns
`narrative`. Add the explicit `return narrative` line that replaces the
deleted tail:

```python
    if narrative is None:
        return None
    return narrative
```

- [ ] **Step 7: Modify `src/auto_research/extract/workers/ten_k.py` — drop unused module docstring lines and lazy bs4 import (if present)**

Open the module docstring at the top (lines 3–30). Remove the bullet
that begins:

```
3. **Item 8 financials.** Independent of the narrative path: …
```

through the end of that paragraph, so the docstring only enumerates
the surviving narrative single-shot and narrative RAG paths.

The `from bs4 import BeautifulSoup` import lives INSIDE
`_render_table_html_to_text` (already removed in Step 5). Verify it is
not imported at module top — grep:

```
grep -n "bs4\|BeautifulSoup" src/auto_research/extract/workers/ten_k.py
```
Expected: no output.

- [ ] **Step 8: Modify `src/auto_research/extract/schemas.py` — drop classes and field**

Remove the `FinancialLineItem` class (currently `schemas.py:179–193`),
the `TenKFinancials` class (currently `schemas.py:196–224`), and the
`TenKOutput.financials` field with its surrounding comment (currently
`schemas.py:248–254`). After the edit, `TenKOutput` ends at
`risk_factor_deltas` and the `# --- Per-field 10-K narrative partials`
section header follows immediately.

In the `__all__` tuple at the bottom of the file, remove the
`"FinancialLineItem"` line and the `"TenKFinancials"` line.

- [ ] **Step 9: Modify `src/auto_research/extract/prompts/ten_k_narrative.py` — drop the "do not populate financials" instruction**

In `TEN_K_NARRATIVE_PROMPT`, delete this Constraints-block line
(currently around `ten_k_narrative.py:114`):

```
- DO NOT populate `financials` (Item 8 is handled by a separate prompt).
```

Update the example block at lines 72–73 — change the example's
preamble comment from:

```
Example of a fully-formed narrative TenKOutput (financials and
language_novelty_score omitted — see Constraints):
```

to:

```
Example of a fully-formed narrative TenKOutput (language_novelty_score
omitted — see Constraints):
```

Do NOT bump `TEN_K_NARRATIVE_PROMPT_VERSION` — per the project's
pre-deployment policy, no version bump until a downstream worker
consumes the contract.

- [ ] **Step 10: Modify `tests/unit/test_models.py` — drop the financials routing test**

Delete the test at `test_models.py:50–53`:

```python
def test_route_model_routes_ten_k_financials_to_haiku() -> None:
    # 10-K Item 8 financials: table → JSON is a templated, high-volume
    # pattern-recognition task per §7.3 — Haiku, not Sonnet.
    assert route_model("ten_k", "financials") == "claude-haiku-4-5"
```

- [ ] **Step 11: Modify `tests/unit/test_extract_prompts.py` — drop the financials import, test, and stale assertion**

Delete the import block at `test_extract_prompts.py:19–22`:

```python
from auto_research.extract.prompts.ten_k_financials import (
    TEN_K_FINANCIALS_PROMPT,
    TEN_K_FINANCIALS_PROMPT_VERSION,
)
```

Delete the `test_ten_k_financials_prompt_exports` test (currently at
`test_extract_prompts.py:120–130`) and its preceding section header
comment.

Delete the stale assertion at `test_extract_prompts.py:113`:

```python
    assert "financials" in TEN_K_NARRATIVE_PROMPT  # mentioned as "do not populate"
```

The surrounding `test_ten_k_narrative_prompt_exports` test continues
to exist and continues to assert the five existing narrative field
names; we add new assertions for the three new fields in a later task.

- [ ] **Step 12: Modify `tests/unit/test_extract_schemas.py` — drop the financials imports, test block, and update `TenKOutput` constructors**

Delete `FinancialLineItem` (`test_extract_schemas.py:26`) and
`TenKFinancials` (`test_extract_schemas.py:31`) from the import block.

Delete the entire TenKFinancials/FinancialLineItem section starting at
the section-header comment `# --- TenKFinancials + FinancialLineItem
(Item 8 structured extraction) ---` (currently
`test_extract_schemas.py:429`) through the last test in that block
(`test_ten_k_output_accepts_financials_when_supplied`, last line ~547).

Delete the test `test_ten_k_output_financials_defaults_to_none`
(lines 492–496).

The `test_ten_k_output_language_novelty_score_defaults_to_zero` test
(lines 499–517) constructs a full `TenKOutput`. Leave it for now; it
will be updated when we add the three new fields in later tasks.

- [ ] **Step 13: Modify `tests/unit/test_extract_worker_ten_k.py` — drop financials imports, helper, and Item 8 tests**

Update the schemas import at `test_extract_worker_ten_k.py:28–32`
from:

```python
from auto_research.extract.schemas import (
    Citation,
    FinancialLineItem,
    TenKFinancials,
)
```

to:

```python
from auto_research.extract.schemas import Citation
```

Update the worker import at `test_extract_worker_ten_k.py:33–37` from:

```python
from auto_research.extract.workers.ten_k import (
    _merge_financials,
    _render_table_html_to_text,
    extract_ten_k,
)
```

to:

```python
from auto_research.extract.workers.ten_k import extract_ten_k
```

Delete the `_valid_financials` helper (`test_extract_worker_ten_k.py:77–97`).

Delete the assertion `assert out.financials is None` at lines 168,
188, and 338 — these live in `test_ten_k_single_shot_branch_no_chunkset`,
`test_ten_k_single_shot_branch_short_doc_with_narrative_chunkset`, and
`test_ten_k_rag_branch_fires_above_cutoff`. The surrounding tests
remain and continue to exercise their branch-coverage purpose.

Delete `test_ten_k_item8_financials_extracted_from_table_html`
(lines 497–524).

Delete `test_render_table_html_strips_tags_and_bridges_cells`
(lines 527–538).

Delete `test_merge_financials_takes_first_non_none_per_field`
(lines 544–end of test, ~600). Remove the section header comment
`# --- Item 8 path ---` that precedes it.

If after these deletions the `import json` at the top of the test
file is unused, drop it; the surviving tests do still use `json` in
`test_ten_k_rag_identity_disagreement_quarantines`, so it stays.

- [ ] **Step 14: Run the lint+typecheck+test gate**

```
make check
```

Expected: all green. `pyright`/`mypy` cleanly resolves the absence of
the deleted symbols; `pytest tests/unit tests/feast -m "not
broad_fixture"` reports zero failures (the surviving tests are
self-contained against the narrative paths).

- [ ] **Step 15: Re-run the sweep**

```
rg -n "TenKFinancials|FinancialLineItem|TEN_K_FINANCIALS|_extract_item8_financials|_merge_financials|_render_table_html_to_text" -tpy src/ tests/
```

Expected: zero matches.

- [ ] **Step 16: Commit**

```bash
git add -u src/ tests/
git rm src/auto_research/extract/prompts/ten_k_financials.py
git commit -m "$(cat <<'EOF'
refactor(extract): drop LLM item 8 financials path — superseded by XBRL

The 2026-05-28 XBRL SQL layer spec covers every line item this path
extracted; LLM table extraction is now pure duplication. Removes:

- TenKFinancials + FinancialLineItem schemas + TenKOutput.financials
- TEN_K_FINANCIALS_PROMPT + prompt-version constant
- _extract_item8_financials, _merge_financials,
  _render_table_html_to_text worker helpers + Item 8 branch in
  extract_ten_k
- ("ten_k", "financials") routing row
- "do not populate financials" instruction from TEN_K_NARRATIVE_PROMPT
- All associated unit tests

Pre-deployment cleanup: no consumers, no SCHEMA_VERSION bump per
the prompt-version policy.

Re-scope of #79; spec: docs/superpowers/specs/2026-05-29-tenk-narrative-financial-disclosures-design.md
EOF
)"
```

Then verify the commit:

```
git status && git log --oneline -1
```
Expected: working tree clean (only `learning/` untracked files), HEAD
is the new deletion commit.

---

## Task 3: Add `going_concern: Claim | None` end-to-end (TDD)

**Goal:** Add the first of three narrative-only fields. Each step is a
small TDD cycle: failing test, minimal impl, passing test, then move on.
After this task, `going_concern` is a required field on `TenKOutput`,
extracted by both the single-shot and RAG paths, with full unit-test
coverage. One commit.

**Files:**
- Modify: `src/auto_research/extract/schemas.py`
- Modify: `src/auto_research/_models.py`
- Modify: `src/auto_research/extract/prompts/ten_k_narrative_field.py`
- Modify: `src/auto_research/extract/prompts/ten_k_narrative.py`
- Modify: `tests/unit/test_models.py`
- Modify: `tests/unit/test_extract_schemas.py`
- Modify: `tests/unit/test_extract_prompts.py`
- Modify: `tests/unit/test_extract_worker_ten_k.py`

- [ ] **Step 1: Write the failing partial-schema test**

Add to `tests/unit/test_extract_schemas.py`, immediately before the
`# --- Per-worker outputs --` section header (or wherever the file's
narrative-partials block lives — search for `TenKAccrualFlagsPartial`
for the canonical anchor):

```python
def test_ten_k_going_concern_partial_carries_identity_and_field() -> None:
    """`TenKGoingConcernPartial` is the RAG-path schema for the new
    going_concern field — same identity-fields + single-narrative-field
    shape as the other TenK*Partial models."""
    from auto_research.extract.schemas import TenKGoingConcernPartial

    p = TenKGoingConcernPartial(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        going_concern=_claim(),
    )
    assert p.going_concern is not None
    assert p.going_concern.confidence == "medium"


def test_ten_k_going_concern_partial_accepts_none() -> None:
    """`going_concern` is `Claim | None` — the modal case in
    `universe_v1` is the auditor's unqualified opinion, where the
    field MUST be None rather than a fabricated Claim."""
    from auto_research.extract.schemas import TenKGoingConcernPartial

    p = TenKGoingConcernPartial(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        going_concern=None,
    )
    assert p.going_concern is None
```

- [ ] **Step 2: Run the test to verify it fails**

```
uv run pytest tests/unit/test_extract_schemas.py::test_ten_k_going_concern_partial_carries_identity_and_field -v
```

Expected: FAIL with `ImportError: cannot import name 'TenKGoingConcernPartial'`.

- [ ] **Step 3: Add `TenKGoingConcernPartial` to `schemas.py`**

In `src/auto_research/extract/schemas.py`, find the existing
`TenKAccrualFlagsPartial` class. Add the new partial AFTER it (so the
file order matches the field-config order added in Step 7), using the
exact same shape:

```python
class TenKGoingConcernPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    going_concern: Claim | None
```

Add `"TenKGoingConcernPartial"` to the `__all__` tuple at the bottom
of the file, keeping the tuple alphabetized to match the existing
convention.

- [ ] **Step 4: Run the test to verify it passes**

```
uv run pytest tests/unit/test_extract_schemas.py::test_ten_k_going_concern_partial_carries_identity_and_field tests/unit/test_extract_schemas.py::test_ten_k_going_concern_partial_accepts_none -v
```

Expected: both PASS.

- [ ] **Step 5: Write the failing routing-table test**

In `tests/unit/test_models.py`, after the existing
`test_route_model_returns_haiku_for_routine_extraction` test, add:

```python
def test_route_model_routes_ten_k_going_concern_to_haiku() -> None:
    """Going-concern is binary auditor language — templated
    pattern-recognition tier per spec §7.3 ⇒ Haiku."""
    assert route_model("ten_k", "going_concern") == "claude-haiku-4-5"
```

- [ ] **Step 6: Run the test to verify it fails**

```
uv run pytest tests/unit/test_models.py::test_route_model_routes_ten_k_going_concern_to_haiku -v
```

Expected: FAIL with `ValueError: no model routed for (worker='ten_k', task='going_concern')`.

- [ ] **Step 7: Add the routing row**

In `src/auto_research/_models.py`, in the `_ROUTING` dict's 10-K
section (immediately after the existing 10-K narrative rows,
`risk_factor_deltas` line), add:

```python
    # Going-concern, ICFR material weaknesses, critical-accounting-
    # estimate changes: narrative-only signals XBRL cannot give;
    # templated language-pattern recognition per spec §7.3 ⇒ Haiku.
    ("ten_k", "going_concern"): _HAIKU,
```

(The other two routing rows will land in Tasks 4 and 5 under the same
comment header.)

- [ ] **Step 8: Run the routing test to verify it passes**

```
uv run pytest tests/unit/test_models.py::test_route_model_routes_ten_k_going_concern_to_haiku -v
```

Expected: PASS.

- [ ] **Step 9: Write the failing RAG-config test**

In `tests/unit/test_extract_prompts.py`, in the section that already
covers `TEN_K_NARRATIVE_FIELD_CONFIGS` (search for the existing
import of `TEN_K_NARRATIVE_FIELD_CONFIGS` or the related test; if no
such test exists yet, add one immediately after
`test_ten_k_narrative_prompt_exports`):

```python
def test_ten_k_narrative_field_configs_includes_going_concern() -> None:
    """The RAG per-field loop iterates TEN_K_NARRATIVE_FIELD_CONFIGS.
    A new field MUST appear there with a non-empty retrieval_query that
    names its source section (so retrieval drift is mechanically
    catchable) and a description that names the source section too."""
    from auto_research.extract.prompts.ten_k_narrative_field import (
        TEN_K_NARRATIVE_FIELD_CONFIGS,
    )

    by_name = {c.field_name: c for c in TEN_K_NARRATIVE_FIELD_CONFIGS}
    assert "going_concern" in by_name
    config = by_name["going_concern"]
    assert config.retrieval_query.strip(), "retrieval_query is empty"
    assert "Item 8" in config.retrieval_query or "going concern" in config.retrieval_query.lower()
    assert "Item 8" in config.description or "going concern" in config.description.lower()
```

- [ ] **Step 10: Run the test to verify it fails**

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_field_configs_includes_going_concern -v
```

Expected: FAIL with `AssertionError: assert 'going_concern' in {...}`.

- [ ] **Step 11: Add the RAG field config**

In `src/auto_research/extract/prompts/ten_k_narrative_field.py`:

(a) Extend the schemas import to include the new partial:

```python
from auto_research.extract.schemas import (
    TenKAccrualFlagsPartial,
    TenKCustomerMentionsPartial,
    TenKGoingConcernPartial,
    TenKGuidanceTonePartial,
    TenKRiskFactorDeltasPartial,
    TenKSupplierMentionsPartial,
)
```

(b) Append a new `TenKNarrativeFieldConfig` entry at the END of the
existing `TEN_K_NARRATIVE_FIELD_CONFIGS` tuple (after
`risk_factor_deltas`, before the closing paren), preserving the
"new fields go at the end" convention:

```python
    TenKNarrativeFieldConfig(
        field_name="going_concern",
        schema=TenKGoingConcernPartial,
        description=(
            "A single Claim quoting verbatim the auditor's "
            "'substantial doubt' sentence from the Item 8 audit report "
            "or the Item 7 liquidity discussion, or null when the audit "
            "report carries an unqualified opinion. Do NOT paraphrase — "
            "quote the actual disclaimer sentence."
        ),
        retrieval_query=(
            "Does the auditor's report in Item 8 or the liquidity "
            "discussion in Item 7 express substantial doubt about the "
            "company's ability to continue as a going concern?"
        ),
    ),
```

- [ ] **Step 12: Run the RAG-config test to verify it passes**

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_field_configs_includes_going_concern -v
```

Expected: PASS.

- [ ] **Step 13: Write the failing `TenKOutput` field test**

In `tests/unit/test_extract_schemas.py`, add (near the other
`TenKOutput` tests):

```python
def test_ten_k_output_carries_going_concern_field() -> None:
    """`TenKOutput.going_concern: Claim | None` — required field
    (no default) consistent with the existing narrative fields
    (guidance_tone, accrual_flags, ...). Both narrative paths
    (single-shot + RAG) are responsible for populating it."""
    out = TenKOutput(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        guidance_tone=_claim(),
        accrual_flags=[],
        supplier_mentions=[],
        customer_mentions=[],
        risk_factor_deltas=[],
        going_concern=_claim(),
        icfr_material_weaknesses=[],
        critical_accounting_estimate_changes=[],
    )
    assert out.going_concern is not None
    assert out.going_concern.confidence == "medium"


def test_ten_k_output_going_concern_accepts_none() -> None:
    out = TenKOutput(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        guidance_tone=_claim(),
        accrual_flags=[],
        supplier_mentions=[],
        customer_mentions=[],
        risk_factor_deltas=[],
        going_concern=None,
        icfr_material_weaknesses=[],
        critical_accounting_estimate_changes=[],
    )
    assert out.going_concern is None
```

These tests reference `icfr_material_weaknesses` and
`critical_accounting_estimate_changes` — the fields land in Tasks 4
and 5; these tests will keep passing because TDD requires them to be
written WITH the construction sites all updated together, which we do
in the next step.

- [ ] **Step 14: Run the test to verify it fails**

```
uv run pytest tests/unit/test_extract_schemas.py::test_ten_k_output_carries_going_concern_field -v
```

Expected: FAIL — either `TenKOutput.__init__() got unexpected keyword
argument 'going_concern'` (before the field exists) or a `pydantic`
validation error for missing required fields (after the field exists
but the constructor is still missing the other two not-yet-added
fields). Either way, FAIL.

- [ ] **Step 15: Add all three fields to `TenKOutput` at once**

This is the only place in the plan where Task 3 touches the surface
area that Tasks 4 and 5 also need. Adding fields one at a time would
break every `TenKOutput(...)` construction site mid-task; adding all
three together is the only way to keep `make check` green between
tasks.

In `src/auto_research/extract/schemas.py`, find the `TenKOutput` class.
After the `risk_factor_deltas: list[RiskFactorDelta]` line, add:

```python
    # Narrative-only signals XBRL definitionally cannot give. Both
    # narrative paths (single-shot via TEN_K_NARRATIVE_PROMPT, RAG via
    # the per-field config loop) populate these. Spec:
    # docs/superpowers/specs/2026-05-29-tenk-narrative-financial-disclosures-design.md.
    going_concern: Claim | None
    icfr_material_weaknesses: list[Claim]
    critical_accounting_estimate_changes: list[Claim]
```

The existing `language_novelty_score: float = 0.0` field stays where
it is (the narrative prompt still leaves it for downstream
computation). Do NOT bump `TenKOutput.SCHEMA_VERSION`.

- [ ] **Step 16: Update every surviving `TenKOutput(...)` construction site**

This is the step that fans out across the test files. Each surviving
`TenKOutput(...)` constructor must now pass the three new fields.

Search for construction sites:

```
rg -n "TenKOutput\(" tests/ src/
```

For each test that constructs a `TenKOutput`, add these three kwargs
(with `_claim()` or `None` as appropriate to the test's intent):

```python
        going_concern=None,
        icfr_material_weaknesses=[],
        critical_accounting_estimate_changes=[],
```

Specific anchor: `tests/unit/test_extract_schemas.py:499` —
`test_ten_k_output_language_novelty_score_defaults_to_zero` — add the
three kwargs before the closing paren.

Source-code construction sites: there are none today. The
production-path `TenKOutput(...)` constructions live in
`_extract_ten_k_rag` (`workers/ten_k.py`); that constructor uses
`**narrative_partials` so it automatically picks up the new keys once
they exist in `narrative_partials` (which happens once Tasks 3/4/5's
field configs land — until then the RAG path would fail).

Per the same logic, the single-shot path's
`run_single_shot_extraction(... output_model=TenKOutput)` would fail
if the LLM response (now matching the updated prompt) does not include
the three new fields. The prompt update in Step 22 below handles this.

- [ ] **Step 17: Update the worker's single-shot test fixtures**

In `tests/unit/test_extract_worker_ten_k.py`, find `_valid_narrative`
(currently `test_extract_worker_ten_k.py:52`). It returns a dict that
gets fed through a mocked Anthropic response into
`run_single_shot_extraction` and parsed as `TenKOutput`. Add the three
new fields to that dict, after `risk_factor_deltas`:

```python
        "going_concern": None,
        "icfr_material_weaknesses": [],
        "critical_accounting_estimate_changes": [],
```

The `_rag_partials_in_order` helper (currently
`test_extract_worker_ten_k.py:357`) returns five dicts — one per
partial schema. We extend it in Tasks 4–5 (and finalize in Task 5)
because the RAG worker iterates ALL field configs. For now (Task 3),
we extend it for `going_concern` only AFTER the field-config addition
in Step 11. Concretely, after the five existing dicts, append:

```python
        {
            **base,
            "going_concern": {
                "citation": {"source_quote": quote},
                "confidence": "high",
            },
        },
```

That keeps the test list aligned with `TEN_K_NARRATIVE_FIELD_CONFIGS`
which now also has six entries (Tasks 4 and 5 each grow this list by
one more).

- [ ] **Step 17b: Extend the two existing RAG tests that hard-code the field count**

After Task 3, the worker iterates 6 configs, but
`test_ten_k_rag_branch_fires_above_cutoff` and
`test_ten_k_rag_identity_disagreement_quarantines` both feed exactly
5 mocked responses and assert `call_count == 5`. The fake-client
sequence will exhaust on the 6th call. Update both tests:

In `test_ten_k_rag_branch_fires_above_cutoff` (currently
`test_extract_worker_ten_k.py:195`):

(a) Extend `field_to_keyword` with an entry for the new field:

```python
    field_to_keyword = {
        "guidance_tone": "growth",
        "accrual_flags": "accrual",
        "supplier_mentions": "supplier",
        "customer_mentions": "customer",
        "risk_factor_deltas": "risk",
        "going_concern": "going concern",
    }
```

(b) Extend `_response_for(field)` to handle `going_concern`. After
the `if field == "risk_factor_deltas":` block, before the `raise
ValueError`, add:

```python
        if field == "going_concern":
            return {
                **base,
                "going_concern": {
                    "citation": {"source_quote": sentinel},
                    "confidence": "high",
                },
            }
```

(c) Extend `responses_in_order`'s field tuple to include
`going_concern` at the end:

```python
    responses_in_order = [
        _response_for(f)
        for f in (
            "guidance_tone",
            "accrual_flags",
            "supplier_mentions",
            "customer_mentions",
            "risk_factor_deltas",
            "going_concern",
        )
    ]
```

(d) Update the assertions:

```python
    assert len(queries_seen) == 6  # one query per narrative field
    assert client.messages.create.call_count == 6  # type: ignore[attr-defined]
```

In `test_ten_k_rag_identity_disagreement_quarantines` (currently
`test_extract_worker_ten_k.py:440`), the test calls
`_rag_partials_in_order` which after Step 17 returns 6 dicts. Update
the response-sequence line and the call_count assert:

```python
    client = _fake_client_sequence(
        [base[0], diverged[1], base[2], base[3], base[4], base[5]]
    )
```

```python
    assert client.messages.create.call_count == 6  # type: ignore[attr-defined]
```

- [ ] **Step 18: Run `TenKOutput` tests to verify**

```
uv run pytest tests/unit/test_extract_schemas.py -v
```

Expected: all green, including the two new
`going_concern` tests.

- [ ] **Step 19: Write the failing single-shot prompt test**

In `tests/unit/test_extract_prompts.py`, extend
`test_ten_k_narrative_prompt_exports`. Replace the field-name loop with
one that also includes the three new fields:

```python
    for field in (
        "guidance_tone",
        "accrual_flags",
        "supplier_mentions",
        "customer_mentions",
        "risk_factor_deltas",
        "going_concern",
        "icfr_material_weaknesses",
        "critical_accounting_estimate_changes",
    ):
        assert field in TEN_K_NARRATIVE_PROMPT, f"missing instruction for {field}"
```

- [ ] **Step 20: Run the test to verify it fails**

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_prompt_exports -v
```

Expected: FAIL with `AssertionError: missing instruction for
going_concern`.

- [ ] **Step 21: Update `TEN_K_NARRATIVE_PROMPT` for `going_concern`**

In `src/auto_research/extract/prompts/ten_k_narrative.py`, in the
"Fields to populate:" block (currently lines 32–61), append after the
`risk_factor_deltas:` bullet (and before the schema-shape paragraph):

```
- going_concern: a single Claim quoting verbatim the auditor's
  "substantial doubt" sentence from the Item 8 audit report or the
  Item 7 liquidity discussion, or null when the audit report carries
  an unqualified opinion. Do NOT paraphrase — quote the actual
  disclaimer sentence. Confidence categorical
  ("high", "medium", "low").
```

(`icfr_material_weaknesses` and `critical_accounting_estimate_changes`
bullets are added in Tasks 4 and 5.)

In the same file, update the example JSON object (lines 75–100) to
include `"going_concern": null` after `"risk_factor_deltas": []`. Use
`null` (not an example Claim) so the example doesn't mislead the model
into thinking going-concern is the modal outcome.

Do NOT bump `TEN_K_NARRATIVE_PROMPT_VERSION`.

- [ ] **Step 22: Run the single-shot prompt test to verify it passes**

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_prompt_exports -v
```

Expected: PASS.

- [ ] **Step 23: Write the failing RAG-worker happy-path test**

In `tests/unit/test_extract_worker_ten_k.py`, add to the RAG section
(after `test_ten_k_rag_branch_fires_above_cutoff`):

```python
def test_ten_k_rag_populates_going_concern_when_planted(
    tmp_path: Path,
) -> None:
    """When the retrieved Item 8 audit passage contains a substantial-
    doubt sentence, the going_concern field on the merged TenKOutput
    is a Claim quoting that sentence."""
    long_raw = "word " * 200_000
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-going-001",
    )
    going_concern_text = (
        "the conditions raise substantial doubt about the Company's "
        "ability to continue as a going concern."
    )
    parent_text = f"Item 8 audit report. {going_concern_text}"
    parent = ParentChunk(
        text=parent_text,
        section_name="item_8",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    base = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    # Each per-field response cites a quote from `parent_text` so the
    # source-quote resolver inside _common matches cleanly. For the
    # five existing fields, an empty list / minimal valid value
    # suffices.
    responses = [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": "audit report"},
                "confidence": "medium",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
        {
            **base,
            "going_concern": {
                "citation": {"source_quote": going_concern_text},
                "confidence": "high",
            },
        },
    ]
    client = _fake_client_sequence(responses)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-going-001",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: [parent],
    )
    assert out is not None
    assert out.going_concern is not None
    assert "substantial doubt" in out.going_concern.citation.source_quote
    assert out.going_concern.confidence == "high"
    assert client.messages.create.call_count == 6  # type: ignore[attr-defined]
```

```python
def test_ten_k_rag_going_concern_absent_returns_none(
    tmp_path: Path,
) -> None:
    """Unqualified audit opinion → partial returns going_concern=None
    → merged TenKOutput carries None."""
    long_raw = "word " * 200_000
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-going-002",
    )
    parent_text = (
        "Item 8 audit report. In our opinion, the financial statements "
        "present fairly, in all material respects, the financial "
        "position of the Company."
    )
    parent = ParentChunk(
        text=parent_text,
        section_name="item_8",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    base = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    responses = [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": "Company"},
                "confidence": "low",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
        {**base, "going_concern": None},
    ]
    client = _fake_client_sequence(responses)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-going-002",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: [parent],
    )
    assert out is not None
    assert out.going_concern is None
```

- [ ] **Step 24: Run the RAG worker tests to verify they fail**

```
uv run pytest tests/unit/test_extract_worker_ten_k.py::test_ten_k_rag_populates_going_concern_when_planted tests/unit/test_extract_worker_ten_k.py::test_ten_k_rag_going_concern_absent_returns_none -v
```

Expected: prior to the field-config landing (Step 11), the worker
would not call the 6th LLM and the assertion `call_count == 6` would
fail. By this step, the config exists and the test should PASS. If
the test fails, the failure points to the worker not iterating the
new field — debug from there.

- [ ] **Step 25: Run `make check` once for the whole task**

```
make check
```

Expected: all green.

- [ ] **Step 26: Commit**

```bash
git add -u src/ tests/
git commit -m "$(cat <<'EOF'
feat(extract): add going_concern narrative field to 10-K worker

XBRL covers the structured financial line items; this is the first
of three narrative-only signals XBRL definitionally can't give. The
auditor's "substantial doubt" sentence is the binary going-concern
signal.

- TenKGoingConcernPartial (RAG-path partial schema)
- TenKOutput.going_concern: Claim | None (also adds the other two
  new fields' attributes pre-emptively so test construction sites
  don't break mid-task; the other two get their own configs +
  prompts in #4 / #5 follow-up tasks)
- _models routing row (Haiku per §7.3)
- TEN_K_NARRATIVE_FIELD_CONFIGS entry (RAG path)
- TEN_K_NARRATIVE_PROMPT bullet (single-shot path)
- Unit tests: partial-schema construction, routing-table coverage,
  RAG-path positive + absent path

No SCHEMA_VERSION or PROMPT_VERSION bumps per the pre-deployment
policy.

Re-scope of #79; spec: docs/superpowers/specs/2026-05-29-tenk-narrative-financial-disclosures-design.md
EOF
)"
```

---

## Task 4: Add `icfr_material_weaknesses: list[Claim]` end-to-end (TDD)

**Goal:** Add the second of three narrative-only fields. Same pattern
as Task 3. The `TenKOutput` field was already added in Task 3 Step 15
(all three fields land together to keep `make check` green between
tasks), so Task 4 adds the partial schema, the routing row, the RAG
field config, the single-shot prompt bullet, and the tests. One commit.

**Files:**
- Modify: `src/auto_research/extract/schemas.py`
- Modify: `src/auto_research/_models.py`
- Modify: `src/auto_research/extract/prompts/ten_k_narrative_field.py`
- Modify: `src/auto_research/extract/prompts/ten_k_narrative.py`
- Modify: `tests/unit/test_models.py`
- Modify: `tests/unit/test_extract_schemas.py`
- Modify: `tests/unit/test_extract_prompts.py`
- Modify: `tests/unit/test_extract_worker_ten_k.py`

- [ ] **Step 1: Write the failing partial-schema test**

Add to `tests/unit/test_extract_schemas.py` after the going_concern
partial tests:

```python
def test_ten_k_icfr_material_weaknesses_partial_carries_identity_and_field() -> None:
    """`TenKIcfrMaterialWeaknessesPartial` is the RAG-path schema for
    Item 9A material-weakness disclosures."""
    from auto_research.extract.schemas import (
        TenKIcfrMaterialWeaknessesPartial,
    )

    p = TenKIcfrMaterialWeaknessesPartial(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        icfr_material_weaknesses=[_claim(), _claim(confidence="high")],
    )
    assert len(p.icfr_material_weaknesses) == 2
    assert p.icfr_material_weaknesses[1].confidence == "high"


def test_ten_k_icfr_material_weaknesses_partial_accepts_empty_list() -> None:
    """`icfr_material_weaknesses` is `list[Claim]` — the modal case
    in `universe_v1` is ICFR-effective (empty list)."""
    from auto_research.extract.schemas import (
        TenKIcfrMaterialWeaknessesPartial,
    )

    p = TenKIcfrMaterialWeaknessesPartial(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        icfr_material_weaknesses=[],
    )
    assert p.icfr_material_weaknesses == []
```

- [ ] **Step 2: Run the test to verify it fails**

```
uv run pytest tests/unit/test_extract_schemas.py::test_ten_k_icfr_material_weaknesses_partial_carries_identity_and_field -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add `TenKIcfrMaterialWeaknessesPartial` to `schemas.py`**

After `TenKGoingConcernPartial`, add:

```python
class TenKIcfrMaterialWeaknessesPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    icfr_material_weaknesses: list[Claim]
```

Add `"TenKIcfrMaterialWeaknessesPartial"` to `__all__` alphabetically.

- [ ] **Step 4: Run the test to verify it passes**

```
uv run pytest tests/unit/test_extract_schemas.py::test_ten_k_icfr_material_weaknesses_partial_carries_identity_and_field tests/unit/test_extract_schemas.py::test_ten_k_icfr_material_weaknesses_partial_accepts_empty_list -v
```

Expected: PASS.

- [ ] **Step 5: Write the failing routing-table test**

In `tests/unit/test_models.py`, after Task 3's
`test_route_model_routes_ten_k_going_concern_to_haiku`:

```python
def test_route_model_routes_ten_k_icfr_material_weaknesses_to_haiku() -> None:
    """ICFR material-weakness language is Item 9A pattern recognition
    — Haiku per §7.3."""
    assert (
        route_model("ten_k", "icfr_material_weaknesses")
        == "claude-haiku-4-5"
    )
```

- [ ] **Step 6: Run the test to verify it fails**

```
uv run pytest tests/unit/test_models.py::test_route_model_routes_ten_k_icfr_material_weaknesses_to_haiku -v
```

Expected: FAIL with `ValueError: no model routed`.

- [ ] **Step 7: Add the routing row**

In `src/auto_research/_models.py`, immediately after the
`("ten_k", "going_concern")` row added in Task 3, append:

```python
    ("ten_k", "icfr_material_weaknesses"): _HAIKU,
```

- [ ] **Step 8: Run the routing test to verify it passes**

```
uv run pytest tests/unit/test_models.py::test_route_model_routes_ten_k_icfr_material_weaknesses_to_haiku -v
```

Expected: PASS.

- [ ] **Step 9: Write the failing RAG-config test**

In `tests/unit/test_extract_prompts.py`:

```python
def test_ten_k_narrative_field_configs_includes_icfr_material_weaknesses() -> None:
    from auto_research.extract.prompts.ten_k_narrative_field import (
        TEN_K_NARRATIVE_FIELD_CONFIGS,
    )

    by_name = {c.field_name: c for c in TEN_K_NARRATIVE_FIELD_CONFIGS}
    assert "icfr_material_weaknesses" in by_name
    config = by_name["icfr_material_weaknesses"]
    assert config.retrieval_query.strip()
    assert "Item 9A" in config.retrieval_query
    assert "Item 9A" in config.description
```

- [ ] **Step 10: Run the test to verify it fails**

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_field_configs_includes_icfr_material_weaknesses -v
```

Expected: FAIL.

- [ ] **Step 11: Add the RAG field config**

In `src/auto_research/extract/prompts/ten_k_narrative_field.py`, extend
the schemas import (insert `TenKIcfrMaterialWeaknessesPartial`
alphabetically), then append a new
`TenKNarrativeFieldConfig` entry to `TEN_K_NARRATIVE_FIELD_CONFIGS`
after `going_concern`:

```python
    TenKNarrativeFieldConfig(
        field_name="icfr_material_weaknesses",
        schema=TenKIcfrMaterialWeaknessesPartial,
        description=(
            "A list of Claims, one per distinct material weakness "
            "disclosed in management's Item 9A internal-controls-over-"
            "financial-reporting (ICFR) report. Empty list when "
            "management concludes ICFR is effective with no material "
            "weaknesses identified. Quote the verbatim weakness "
            "description sentence (e.g., 'we did not maintain "
            "effective controls over X')."
        ),
        retrieval_query=(
            "Does management's Item 9A internal-controls-over-financial-"
            "reporting report identify any material weaknesses in ICFR?"
        ),
    ),
```

- [ ] **Step 12: Run the RAG-config test to verify it passes**

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_field_configs_includes_icfr_material_weaknesses -v
```

Expected: PASS.

- [ ] **Step 13: Update `TEN_K_NARRATIVE_PROMPT` for `icfr_material_weaknesses`**

In `src/auto_research/extract/prompts/ten_k_narrative.py`, in the
Fields-to-populate block, after the `going_concern:` bullet added in
Task 3, append:

```
- icfr_material_weaknesses: a list of Claims, one per distinct
  material weakness disclosed in management's Item 9A internal-
  controls-over-financial-reporting (ICFR) report. Empty list when
  management concludes ICFR is effective with no material
  weaknesses. Quote the verbatim weakness description.
```

Update the example JSON to include `"icfr_material_weaknesses": []`
after `"going_concern": null`.

Do NOT bump `TEN_K_NARRATIVE_PROMPT_VERSION`.

The `test_ten_k_narrative_prompt_exports` test (already extended in
Task 3 to include `icfr_material_weaknesses` in the field-name loop)
covers this. Re-run:

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_prompt_exports -v
```

Expected: PASS.

- [ ] **Step 14: Update `_rag_partials_in_order` to include the icfr partial**

In `tests/unit/test_extract_worker_ten_k.py`, extend
`_rag_partials_in_order` (currently has 6 entries after Task 3) by
appending after the `going_concern` entry:

```python
        {**base, "icfr_material_weaknesses": []},
```

The list now has 7 entries.

- [ ] **Step 14b: Extend the two existing RAG tests for the new field count**

After Task 4, the worker iterates 7 configs. Update
`test_ten_k_rag_branch_fires_above_cutoff` and
`test_ten_k_rag_identity_disagreement_quarantines` by one entry each:

In `test_ten_k_rag_branch_fires_above_cutoff`:

(a) Append `"icfr_material_weaknesses": "ICFR"` to `field_to_keyword`
(matching the retrieval-query substring added in Step 11).

(b) Extend `_response_for(field)` with:

```python
        if field == "icfr_material_weaknesses":
            return {
                **base,
                "icfr_material_weaknesses": [
                    {
                        "citation": {"source_quote": sentinel},
                        "confidence": "high",
                    }
                ],
            }
```

(c) Append `"icfr_material_weaknesses"` to the field tuple driving
`responses_in_order`.

(d) Update assertions:

```python
    assert len(queries_seen) == 7
    assert client.messages.create.call_count == 7  # type: ignore[attr-defined]
```

In `test_ten_k_rag_identity_disagreement_quarantines`, extend the
response slice and assert:

```python
    client = _fake_client_sequence(
        [base[0], diverged[1], base[2], base[3], base[4], base[5], base[6]]
    )
```

```python
    assert client.messages.create.call_count == 7  # type: ignore[attr-defined]
```

- [ ] **Step 15: Update `_valid_narrative` (single-shot fixture) — no change here**

The Task 3 update already inserted `icfr_material_weaknesses: []` into
the `_valid_narrative` fixture. No edit needed in Task 4.

- [ ] **Step 16: Write the failing RAG-worker happy-path test**

In `tests/unit/test_extract_worker_ten_k.py`, after Task 3's
going-concern tests:

```python
def test_ten_k_rag_populates_icfr_material_weaknesses_when_planted(
    tmp_path: Path,
) -> None:
    """Item 9A material-weakness passage → icfr_material_weaknesses
    on the merged TenKOutput is a non-empty list of Claims."""
    long_raw = "word " * 200_000
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-icfr-001",
    )
    weakness_text = (
        "we did not maintain effective controls over revenue "
        "recognition for the contract modification process."
    )
    parent_text = (
        f"Item 9A. {weakness_text}"
    )
    parent = ParentChunk(
        text=parent_text,
        section_name="item_9a",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    base = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    responses = [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": "controls"},
                "confidence": "low",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
        {**base, "going_concern": None},
        {
            **base,
            "icfr_material_weaknesses": [
                {
                    "citation": {"source_quote": weakness_text},
                    "confidence": "high",
                }
            ],
        },
    ]
    client = _fake_client_sequence(responses)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-icfr-001",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: [parent],
    )
    assert out is not None
    assert len(out.icfr_material_weaknesses) == 1
    assert "effective controls" in out.icfr_material_weaknesses[0].citation.source_quote
    assert out.icfr_material_weaknesses[0].confidence == "high"
    assert client.messages.create.call_count == 7  # type: ignore[attr-defined]
```

```python
def test_ten_k_rag_icfr_effective_returns_empty_list(tmp_path: Path) -> None:
    """ICFR-effective Item 9A → partial returns empty list →
    merged TenKOutput carries an empty list."""
    long_raw = "word " * 200_000
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-icfr-002",
    )
    parent_text = (
        "Item 9A. Management concluded that the Company's internal "
        "control over financial reporting was effective as of "
        "December 31, 2025."
    )
    parent = ParentChunk(
        text=parent_text,
        section_name="item_9a",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    base = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    responses = [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": "Company"},
                "confidence": "low",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
        {**base, "going_concern": None},
        {**base, "icfr_material_weaknesses": []},
    ]
    client = _fake_client_sequence(responses)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-icfr-002",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: [parent],
    )
    assert out is not None
    assert out.icfr_material_weaknesses == []
```

- [ ] **Step 17: Run the RAG worker tests to verify they pass**

```
uv run pytest tests/unit/test_extract_worker_ten_k.py::test_ten_k_rag_populates_icfr_material_weaknesses_when_planted tests/unit/test_extract_worker_ten_k.py::test_ten_k_rag_icfr_effective_returns_empty_list -v
```

Expected: PASS.

- [ ] **Step 18: Run `make check` once for the whole task**

```
make check
```

Expected: green.

- [ ] **Step 19: Commit**

```bash
git add -u src/ tests/
git commit -m "$(cat <<'EOF'
feat(extract): add icfr_material_weaknesses field to 10-K worker

Second of three narrative-only signals XBRL can't give. Captures
management's Item 9A ICFR assessment: list of Claims, one per
material weakness disclosed; empty list when ICFR is effective.

- TenKIcfrMaterialWeaknessesPartial (RAG-path partial schema)
- _models routing row (Haiku per §7.3)
- TEN_K_NARRATIVE_FIELD_CONFIGS entry (RAG path)
- TEN_K_NARRATIVE_PROMPT bullet (single-shot path)
- Unit tests: partial-schema construction, routing-table coverage,
  RAG-path positive + ICFR-effective path

No SCHEMA_VERSION or PROMPT_VERSION bumps per the pre-deployment
policy.

Re-scope of #79; spec: docs/superpowers/specs/2026-05-29-tenk-narrative-financial-disclosures-design.md
EOF
)"
```

---

## Task 5: Add `critical_accounting_estimate_changes: list[Claim]` end-to-end (TDD)

**Goal:** Add the third of three narrative-only fields. Same pattern
as Tasks 3 and 4. The `TenKOutput` field was already added in Task 3
Step 15. One commit.

**Files:**
- Modify: `src/auto_research/extract/schemas.py`
- Modify: `src/auto_research/_models.py`
- Modify: `src/auto_research/extract/prompts/ten_k_narrative_field.py`
- Modify: `src/auto_research/extract/prompts/ten_k_narrative.py`
- Modify: `tests/unit/test_models.py`
- Modify: `tests/unit/test_extract_schemas.py`
- Modify: `tests/unit/test_extract_prompts.py`
- Modify: `tests/unit/test_extract_worker_ten_k.py`

- [ ] **Step 1: Write the failing partial-schema test**

```python
def test_ten_k_critical_accounting_estimate_changes_partial_carries_identity_and_field() -> None:
    """`TenKCriticalAccountingEstimateChangesPartial` is the RAG-path
    schema for Item 7 / Item 8 footnote disclosures of critical
    accounting estimate changes."""
    from auto_research.extract.schemas import (
        TenKCriticalAccountingEstimateChangesPartial,
    )

    p = TenKCriticalAccountingEstimateChangesPartial(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        critical_accounting_estimate_changes=[_claim(confidence="medium")],
    )
    assert len(p.critical_accounting_estimate_changes) == 1


def test_ten_k_critical_accounting_estimate_changes_partial_accepts_empty_list() -> None:
    """Empty list when no YoY change is flagged."""
    from auto_research.extract.schemas import (
        TenKCriticalAccountingEstimateChangesPartial,
    )

    p = TenKCriticalAccountingEstimateChangesPartial(
        cik="0001045810",
        accession_number="0001045810-25-000001",
        fiscal_period_end=date(2025, 1, 31),
        critical_accounting_estimate_changes=[],
    )
    assert p.critical_accounting_estimate_changes == []
```

- [ ] **Step 2: Run the test to verify it fails**

```
uv run pytest tests/unit/test_extract_schemas.py::test_ten_k_critical_accounting_estimate_changes_partial_carries_identity_and_field -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add the partial schema**

After `TenKIcfrMaterialWeaknessesPartial`, add:

```python
class TenKCriticalAccountingEstimateChangesPartial(BaseModel):
    model_config = _FROZEN_STRICT
    SCHEMA_VERSION: ClassVar[str] = "v1"

    cik: str
    accession_number: str
    fiscal_period_end: date
    critical_accounting_estimate_changes: list[Claim]
```

Add `"TenKCriticalAccountingEstimateChangesPartial"` to `__all__`
alphabetically.

- [ ] **Step 4: Run the test to verify it passes**

```
uv run pytest tests/unit/test_extract_schemas.py::test_ten_k_critical_accounting_estimate_changes_partial_carries_identity_and_field tests/unit/test_extract_schemas.py::test_ten_k_critical_accounting_estimate_changes_partial_accepts_empty_list -v
```

Expected: PASS.

- [ ] **Step 5: Write the failing routing-table test**

```python
def test_route_model_routes_ten_k_critical_accounting_estimate_changes_to_haiku() -> None:
    """Critical-estimate language is Item 7 / footnote pattern
    recognition — Haiku per §7.3."""
    assert (
        route_model("ten_k", "critical_accounting_estimate_changes")
        == "claude-haiku-4-5"
    )
```

- [ ] **Step 6: Run the test to verify it fails**

```
uv run pytest tests/unit/test_models.py::test_route_model_routes_ten_k_critical_accounting_estimate_changes_to_haiku -v
```

Expected: FAIL.

- [ ] **Step 7: Add the routing row**

In `src/auto_research/_models.py`, immediately after the
`("ten_k", "icfr_material_weaknesses")` row, append:

```python
    ("ten_k", "critical_accounting_estimate_changes"): _HAIKU,
```

- [ ] **Step 8: Run the routing test to verify it passes**

```
uv run pytest tests/unit/test_models.py::test_route_model_routes_ten_k_critical_accounting_estimate_changes_to_haiku -v
```

Expected: PASS.

- [ ] **Step 9: Write the failing RAG-config test**

```python
def test_ten_k_narrative_field_configs_includes_critical_accounting_estimate_changes() -> None:
    from auto_research.extract.prompts.ten_k_narrative_field import (
        TEN_K_NARRATIVE_FIELD_CONFIGS,
    )

    by_name = {c.field_name: c for c in TEN_K_NARRATIVE_FIELD_CONFIGS}
    assert "critical_accounting_estimate_changes" in by_name
    config = by_name["critical_accounting_estimate_changes"]
    assert config.retrieval_query.strip()
    assert "Item 7" in config.retrieval_query
    assert "Item 7" in config.description
```

- [ ] **Step 10: Run the test to verify it fails**

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_field_configs_includes_critical_accounting_estimate_changes -v
```

Expected: FAIL.

- [ ] **Step 11: Add the RAG field config**

In `src/auto_research/extract/prompts/ten_k_narrative_field.py`, extend
the schemas import (insert
`TenKCriticalAccountingEstimateChangesPartial` alphabetically), then
append the entry to `TEN_K_NARRATIVE_FIELD_CONFIGS` after
`icfr_material_weaknesses`:

```python
    TenKNarrativeFieldConfig(
        field_name="critical_accounting_estimate_changes",
        schema=TenKCriticalAccountingEstimateChangesPartial,
        description=(
            "A list of Claims for accounting estimates that management "
            "flags in Item 7 MD&A 'Critical Accounting Estimates' or "
            "the Item 8 Significant Accounting Policies note as "
            "requiring significant judgment AND where management "
            "indicates a change versus the prior year (new estimate, "
            "methodology change, materially different assumptions). "
            "Empty list when no YoY change is flagged. Quote the "
            "verbatim change-indicating sentence."
        ),
        retrieval_query=(
            "Which critical accounting estimates does Item 7 MD&A or "
            "the Item 8 footnotes flag as new, changed, or requiring "
            "materially different assumptions versus the prior year?"
        ),
    ),
```

- [ ] **Step 12: Run the RAG-config test to verify it passes**

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_field_configs_includes_critical_accounting_estimate_changes -v
```

Expected: PASS.

- [ ] **Step 13: Update `TEN_K_NARRATIVE_PROMPT` for the third field**

In `src/auto_research/extract/prompts/ten_k_narrative.py`, in the
Fields-to-populate block, after the `icfr_material_weaknesses:` bullet
added in Task 4, append:

```
- critical_accounting_estimate_changes: a list of Claims for
  accounting estimates that management flags in Item 7 MD&A "Critical
  Accounting Estimates" or the Item 8 Significant Accounting Policies
  note as requiring significant judgment AND where management
  indicates a change versus the prior year (new estimate, methodology
  change, materially different assumptions). Empty list when no YoY
  change is flagged.
```

Update the example JSON to include
`"critical_accounting_estimate_changes": []` after
`"icfr_material_weaknesses": []`.

Do NOT bump `TEN_K_NARRATIVE_PROMPT_VERSION`.

Re-run the prompt-shape test:

```
uv run pytest tests/unit/test_extract_prompts.py::test_ten_k_narrative_prompt_exports -v
```

Expected: PASS.

- [ ] **Step 14: Update `_rag_partials_in_order` for the third field**

In `tests/unit/test_extract_worker_ten_k.py`, append the third new
entry to `_rag_partials_in_order` after the
`icfr_material_weaknesses` entry:

```python
        {**base, "critical_accounting_estimate_changes": []},
```

The list now has 8 entries — matching the 8 narrative-field configs
in `TEN_K_NARRATIVE_FIELD_CONFIGS` (5 original + 3 new).

- [ ] **Step 14b: Extend the two existing RAG tests for the new field count**

After Task 5, the worker iterates 8 configs. Update
`test_ten_k_rag_branch_fires_above_cutoff` and
`test_ten_k_rag_identity_disagreement_quarantines` one more time:

In `test_ten_k_rag_branch_fires_above_cutoff`:

(a) Append `"critical_accounting_estimate_changes": "critical accounting"`
to `field_to_keyword` (matching the retrieval-query substring added
in Step 11).

(b) Extend `_response_for(field)` with:

```python
        if field == "critical_accounting_estimate_changes":
            return {
                **base,
                "critical_accounting_estimate_changes": [
                    {
                        "citation": {"source_quote": sentinel},
                        "confidence": "high",
                    }
                ],
            }
```

(c) Append `"critical_accounting_estimate_changes"` to the field tuple
driving `responses_in_order`.

(d) Update assertions:

```python
    assert len(queries_seen) == 8
    assert client.messages.create.call_count == 8  # type: ignore[attr-defined]
```

In `test_ten_k_rag_identity_disagreement_quarantines`, extend the
response slice and assert:

```python
    client = _fake_client_sequence(
        [base[0], diverged[1], base[2], base[3], base[4], base[5], base[6], base[7]]
    )
```

```python
    assert client.messages.create.call_count == 8  # type: ignore[attr-defined]
```

- [ ] **Step 15: Update `_valid_narrative` (single-shot fixture) — no change here**

The Task 3 update already inserted
`critical_accounting_estimate_changes: []` into the `_valid_narrative`
fixture.

- [ ] **Step 16: Write the failing RAG-worker happy-path test**

```python
def test_ten_k_rag_populates_critical_accounting_estimate_changes_when_planted(
    tmp_path: Path,
) -> None:
    """Item 7 'Critical Accounting Estimates' passage with a flagged
    YoY change → critical_accounting_estimate_changes is non-empty."""
    long_raw = "word " * 200_000
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-est-001",
    )
    change_text = (
        "we revised our estimated useful life of server hardware from "
        "four to six years effective fiscal 2025."
    )
    parent_text = f"Item 7. Critical Accounting Estimates. {change_text}"
    parent = ParentChunk(
        text=parent_text,
        section_name="item_7",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    base = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    responses = [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": "fiscal 2025"},
                "confidence": "low",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
        {**base, "going_concern": None},
        {**base, "icfr_material_weaknesses": []},
        {
            **base,
            "critical_accounting_estimate_changes": [
                {
                    "citation": {"source_quote": change_text},
                    "confidence": "high",
                }
            ],
        },
    ]
    client = _fake_client_sequence(responses)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-est-001",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: [parent],
    )
    assert out is not None
    assert len(out.critical_accounting_estimate_changes) == 1
    assert (
        "revised our estimated useful life"
        in out.critical_accounting_estimate_changes[0].citation.source_quote
    )
    assert client.messages.create.call_count == 8  # type: ignore[attr-defined]
```

```python
def test_ten_k_rag_critical_estimates_unchanged_returns_empty_list(
    tmp_path: Path,
) -> None:
    """Item 7 passage discusses critical estimates without flagging
    a YoY change → empty list."""
    long_raw = "word " * 200_000
    meta = ChunkMetadata(
        ticker="ACME",
        filing_date=date(2026, 1, 30),
        fiscal_period="FY2025",
        doc_type="10-K",
        doc_id="10k-est-002",
    )
    parent_text = (
        "Item 7. Critical Accounting Estimates. Our estimates for "
        "revenue recognition, inventory valuation, and income taxes "
        "involve significant judgment. There have been no material "
        "changes from the prior year's methodology or assumptions."
    )
    parent = ParentChunk(
        text=parent_text,
        section_name="item_7",
        char_span=(0, len(parent_text)),
        token_count=10,
        table_html=None,
        metadata=meta,
    )
    chunkset = ChunkSet(parents=(parent,), children=())

    base = {
        "cik": "0000000001",
        "accession_number": "0000000001-26-000001",
        "fiscal_period_end": "2025-12-31",
    }
    responses = [
        {
            **base,
            "guidance_tone": {
                "citation": {"source_quote": "judgment"},
                "confidence": "low",
            },
        },
        {**base, "accrual_flags": []},
        {**base, "supplier_mentions": []},
        {**base, "customer_mentions": []},
        {**base, "risk_factor_deltas": []},
        {**base, "going_concern": None},
        {**base, "icfr_material_weaknesses": []},
        {**base, "critical_accounting_estimate_changes": []},
    ]
    client = _fake_client_sequence(responses)
    out = extract_ten_k(
        raw_doc=long_raw,
        doc_id="10k-est-002",
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        anthropic_client=client,
        chunkset=chunkset,
        retrieve_fn=lambda _q: [parent],
    )
    assert out is not None
    assert out.critical_accounting_estimate_changes == []
```

- [ ] **Step 17: Add a single-shot path positive test for the three new fields**

The single-shot path was covered defensively in Task 3 Step 17 (the
`_valid_narrative` fixture got the three new keys with `None`/`[]`
values) but no test exercises a single-shot extraction that POPULATES
them. Add:

```python
def test_ten_k_single_shot_populates_new_narrative_fields(
    tmp_path: Path,
) -> None:
    """Short raw doc → single-shot. The narrative prompt now instructs
    the model to emit going_concern / icfr_material_weaknesses /
    critical_accounting_estimate_changes; verify they round-trip when
    populated (mirrors the existing single-shot narrative branch
    test but proves the new fields work end-to-end on this path too)."""
    raw_doc = (
        "Item 1A. Risk Factors. Our supply chain depends on TSMC.\n"
        "Item 7. Management's Discussion and Analysis. "
        "We expect cautious growth in fiscal 2026.\n"
        "Item 8. Audit report. The conditions raise substantial doubt "
        "about the Company's ability to continue as a going concern.\n"
        "Item 9A. We did not maintain effective controls over revenue "
        "recognition.\n"
        "Critical Accounting Estimates. We revised our estimated "
        "useful life of server hardware from four to six years.\n"
    )
    response = _valid_narrative()
    response["going_concern"] = {
        "citation": {
            "source_quote": (
                "substantial doubt about the Company's ability to "
                "continue as a going concern"
            )
        },
        "confidence": "high",
    }
    response["icfr_material_weaknesses"] = [
        {
            "citation": {
                "source_quote": (
                    "did not maintain effective controls over revenue "
                    "recognition"
                )
            },
            "confidence": "high",
        }
    ]
    response["critical_accounting_estimate_changes"] = [
        {
            "citation": {
                "source_quote": (
                    "revised our estimated useful life of server "
                    "hardware from four to six years"
                )
            },
            "confidence": "medium",
        }
    ]
    client = _fake_client_single(response)
    out = extract_ten_k(
        raw_doc=raw_doc,
        doc_id="10k-singleshot-new",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.going_concern is not None
    assert "substantial doubt" in out.going_concern.citation.source_quote
    assert len(out.icfr_material_weaknesses) == 1
    assert len(out.critical_accounting_estimate_changes) == 1
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]
```

- [ ] **Step 18: Run the RAG and single-shot tests to verify they pass**

```
uv run pytest tests/unit/test_extract_worker_ten_k.py::test_ten_k_rag_populates_critical_accounting_estimate_changes_when_planted tests/unit/test_extract_worker_ten_k.py::test_ten_k_rag_critical_estimates_unchanged_returns_empty_list tests/unit/test_extract_worker_ten_k.py::test_ten_k_single_shot_populates_new_narrative_fields -v
```

Expected: PASS.

- [ ] **Step 19: Run `make check` once for the whole task**

```
make check
```

Expected: green. The full suite now exercises 8 narrative fields on
the RAG path and the new fields on the single-shot path.

- [ ] **Step 20: Commit**

```bash
git add -u src/ tests/
git commit -m "$(cat <<'EOF'
feat(extract): add critical_accounting_estimate_changes to 10-K worker

Third of three narrative-only signals XBRL can't give. Captures
Item 7 / Item 8 footnote disclosures of accounting estimates that
required significant judgment AND changed versus the prior year.

- TenKCriticalAccountingEstimateChangesPartial (RAG-path schema)
- _models routing row (Haiku per §7.3)
- TEN_K_NARRATIVE_FIELD_CONFIGS entry (RAG path)
- TEN_K_NARRATIVE_PROMPT bullet (single-shot path)
- Unit tests: partial-schema construction, routing-table coverage,
  RAG-path positive + unchanged path, single-shot path coverage
  for all three new fields together

No SCHEMA_VERSION or PROMPT_VERSION bumps per the pre-deployment
policy.

Closes #79; spec: docs/superpowers/specs/2026-05-29-tenk-narrative-financial-disclosures-design.md
EOF
)"
```

---

## Task 6: Final verification and PR

**Goal:** Verify the end state and open a PR.

**Files:** none modified.

- [ ] **Step 1: Run the full local gate**

```
make check
```

Expected: ruff + mypy + pytest unit suite all green.

- [ ] **Step 2: Re-run the deletion sweep**

```
rg -n "TenKFinancials|FinancialLineItem|TEN_K_FINANCIALS|_extract_item8_financials|_merge_financials|_render_table_html_to_text" -tpy src/ tests/
```

Expected: zero matches.

- [ ] **Step 3: Verify the three new fields are present on `TenKOutput`**

```
uv run python -c "from auto_research.extract.schemas import TenKOutput; assert 'going_concern' in TenKOutput.model_fields; assert 'icfr_material_weaknesses' in TenKOutput.model_fields; assert 'critical_accounting_estimate_changes' in TenKOutput.model_fields; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Verify the routing-table coverage**

```
uv run python -c "from auto_research._models import route_model; print(route_model('ten_k', 'going_concern')); print(route_model('ten_k', 'icfr_material_weaknesses')); print(route_model('ten_k', 'critical_accounting_estimate_changes'))"
```

Expected: three lines of `claude-haiku-4-5`.

- [ ] **Step 5: Verify the RAG-config order and count**

```
uv run python -c "from auto_research.extract.prompts.ten_k_narrative_field import TEN_K_NARRATIVE_FIELD_CONFIGS; names = [c.field_name for c in TEN_K_NARRATIVE_FIELD_CONFIGS]; print(names); assert names == ['guidance_tone','accrual_flags','supplier_mentions','customer_mentions','risk_factor_deltas','going_concern','icfr_material_weaknesses','critical_accounting_estimate_changes']"
```

Expected: the list printed and no `AssertionError`.

- [ ] **Step 6: Review the diff summary**

```
git diff --stat main..HEAD
```

Expected: roughly equal-sized additions and deletions across `src/`
and `tests/`, no surprise files modified.

- [ ] **Step 7: Push the branch**

```
git push -u origin HEAD
```

If the remote rejects (branch already exists), surface to user before
overwriting.

- [ ] **Step 8: Open the PR**

Per AGENTS.md PR-creation convention:

```bash
gh pr create --title "feat(extract): re-scope #79 — drop LLM item 8, add 3 narrative-only signals" --body "$(cat <<'EOF'
## Summary

Re-scopes #79 in light of the 2026-05-28 XBRL SQL layer spec. The
original 75-field expansion is dropped in favor of:

- Deleting the LLM Item 8 table-extraction path (TenKFinancials,
  FinancialLineItem, TEN_K_FINANCIALS_PROMPT, _extract_item8_financials,
  _merge_financials, _render_table_html_to_text, TenKOutput.financials,
  ("ten_k", "financials") routing row, all matching tests). XBRL is
  authoritative for the line items this path was extracting.
- Adding 3 narrative-only signals XBRL definitionally can't give:
  going_concern: Claim | None, icfr_material_weaknesses: list[Claim],
  critical_accounting_estimate_changes: list[Claim]. Each plugs into
  the existing per-field RAG loop (3 new partials, 3 new
  TenKNarrativeFieldConfig entries, 3 new Haiku routing rows) and into
  the single-shot path (3 new bullets in TEN_K_NARRATIVE_PROMPT).

Net per-filing cost: −~\$0.035/long 10-K. No SCHEMA_VERSION or
PROMPT_VERSION bumps per the pre-deployment prompt-version policy.

Spec: docs/superpowers/specs/2026-05-29-tenk-narrative-financial-disclosures-design.md

Closes #79.

## Test plan

- [ ] `make check` green locally
- [ ] Deletion sweep returns zero matches for the removed symbols
- [ ] Three new fields populated on a planted RAG-path fixture
- [ ] Three new fields populated on a planted single-shot fixture
- [ ] "Absent" tests: going_concern=None when audit is unqualified;
      icfr_material_weaknesses=[] when ICFR is effective;
      critical_accounting_estimate_changes=[] when no YoY change
- [ ] Routing-table coverage tests assert all three new rows route to
      Haiku

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Confirm the PR URL is printed, then post it back to the user.

- [ ] **Step 9: Surface the PR URL to the user.**

The terminal state — user can review, request changes, or merge.

---

## Self-review checklist (for the planner; non-blocking)

- **Spec coverage:** §1 motivation covered by the plan's high-level
  narrative; §2 scope mapped to Tasks 2–5; §3 field semantics drive the
  exact text in the partial schemas (Task 3/4/5 Steps 3) and the prompt
  bullets (Task 3 Step 21, Task 4 Step 13, Task 5 Step 13); §4
  architecture has no code change beyond what Tasks 3–5 do (worker
  iterates configs automatically); §5 deletion checklist is Task 2;
  §6 additions checklist is Tasks 3–5; §7 tests are sprinkled across
  Tasks 3–5; §8 INV compliance enforced (no version bumps); §9 cost
  not in plan (no code action required); §10 sequencing (Task 1
  worktree, Task 6 PR); §11 risks (test fixtures plant the rare
  positive cases per the risk on going-concern + the "absent" branch
  per the risk on critical-estimates subjectivity); §12 acceptance
  criteria covered by Task 6 verification steps.
- **Placeholder scan:** no TBDs, every step has the actual edit shown.
- **Type consistency:** `Claim | None` vs `list[Claim]` field types
  match across the partial schemas, `TenKOutput`, prompt bullets, and
  test fixtures. Partial class names use the same
  `TenKGoingConcernPartial` / `TenKIcfrMaterialWeaknessesPartial` /
  `TenKCriticalAccountingEstimateChangesPartial` casing throughout.
