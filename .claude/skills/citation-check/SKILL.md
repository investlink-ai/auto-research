---
name: citation-check
description: Use when editing extract/guardrails.py, extract/schemas.py, any worker body under extract/, or eval suites that touch hallucination/citation scoring. Enforces INV-2 — every extracted claim must carry a source_span and source_quote that pass post-validation; failures route to data/quarantine/, never silently degrade. The "research discipline" interview story depends on this invariant being mechanically maintained.
allowed-tools: Read, Grep, Bash
---

# citation-check

**Invariant being defended (AGENTS.md INV-2):**

> Every extracted claim carries `source_span: tuple[int, int]` and
> `source_quote: str`. Post-validation asserts
> `source_text[span[0]:span[1]] == source_quote`. Failures route to
> `data/quarantine/{worker}/{doc_id}.json` for human review — never silently
> retried with degraded data.

This is the second most-critical invariant after PIT. It is the mechanical
proof that hallucinations are caught at write-time. The interview story
"every claim traces to a verbatim source quote" is true **only** if this
post-validator is in the failure path of every worker, with no `try/except`
escape hatch.

The natural failure mode under time pressure:

```python
try:
    validated = validate_citation_grounding(output, source_text)
except CitationMismatch:
    logger.warning("citation mismatch", extra={"doc_id": doc.id})
    return output  # INV-2 is now dead. Silent degradation.
```

Sister skill: [[pit-check]] defends the lookahead invariant the same way —
mechanically, with grep patterns, not by hoping the agent remembers.

## When to invoke

Before claiming done on any change that:

- Edits `src/auto_research/extract/guardrails.py` (the validator itself).
- Edits `src/auto_research/extract/schemas.py` (the `Citation` / `Claim`
  base classes — schema changes can silently weaken validation).
- Adds or modifies a worker (`extract/ten_k.py`, `transcript.py`,
  `eight_k.py`, `s_filings.py`).
- Touches `data/quarantine/` paths or the routing logic that writes there.
- Adds or modifies extraction eval (`eval/deepeval_suite.py`,
  `tests/extract/test_*_extraction.py`).

## What "correct" looks like

Guardrails raise; workers route to quarantine; output is never persisted on
failure.

```python
# extract/guardrails.py — CORRECT
class CitationMismatch(ValueError):
    """source_text[span] != source_quote. Hallucination caught."""

def validate_citation_grounding(
    output: BaseModel,
    source_text: str,
) -> None:
    for claim in iter_claims(output):
        actual = source_text[claim.citation.source_span[0]:claim.citation.source_span[1]]
        if actual != claim.citation.source_quote:
            raise CitationMismatch(
                f"span={claim.citation.source_span} "
                f"expected={claim.citation.source_quote!r} "
                f"actual={actual!r}"
            )
```

```python
# extract/ten_k.py — CORRECT worker invocation
def extract_ten_k(raw_doc: RawDoc, prompt_version: str) -> TenKOutput | None:
    output = _call_llm(raw_doc, prompt_version)
    try:
        validate_citation_grounding(output, raw_doc.text)
    except CitationMismatch as exc:
        quarantine_path = (
            DATA_QUARANTINE / "ten_k" / f"{raw_doc.doc_id}.json"
        )
        quarantine_path.write_text(
            QuarantineRecord(
                doc_id=raw_doc.doc_id,
                worker="ten_k",
                prompt_version=prompt_version,
                output=output.model_dump(),
                error=str(exc),
            ).model_dump_json(indent=2)
        )
        return None  # caller checks for None; never persists a quarantined output
    return output
```

```python
# tests/extract/test_citation_grounding.py — required property test
@given(corrupted=corrupted_citation_strategy())
def test_corrupted_citation_routes_to_quarantine(corrupted, tmp_quarantine):
    result = extract_ten_k(corrupted.raw_doc, prompt_version="test")
    assert result is None
    assert (tmp_quarantine / "ten_k" / f"{corrupted.raw_doc.doc_id}.json").exists()
```

## What "wrong" looks like

Any of these patterns means INV-2 is being weakened. Stop and fix.

```python
# WRONG — silent log-and-continue
try:
    validate_citation_grounding(output, source_text)
except CitationMismatch:
    logger.warning("citation mismatch")
    return output

# WRONG — broad except swallows the mismatch
try:
    validate_citation_grounding(output, source_text)
except Exception:
    return output

# WRONG — direct model construction bypasses the validator wrapper
return TenKOutput.model_validate(llm_raw_dict)  # no guardrails call

# WRONG — "soft mode" or "permissive" flag that disables validation
def extract_ten_k(raw_doc, prompt_version, permissive: bool = False):
    output = _call_llm(...)
    if not permissive:
        validate_citation_grounding(output, raw_doc.text)
    return output

# WRONG — heuristic auto-repair instead of quarantine
except CitationMismatch:
    output = _trim_to_matching_substring(output)  # corrupts evidence
    return output

# WRONG — bulk quarantine flush in a cron
def flush_quarantine():
    for f in DATA_QUARANTINE.glob("**/*.json"):
        f.unlink()  # never. quarantine is the audit trail.
```

## Mandatory checks

Run all five. Any miss is blocking.

**1. No swallowed `CitationMismatch` in worker code:**

```bash
rg -nP 'except\s+(CitationMismatch|Exception|ValueError)' src/auto_research/extract/ \
  | rg -v 'quarantine'
# Expected: zero hits. Every except branch must reference a quarantine write
# (or re-raise). If a hit appears, inspect the except body manually.
```

**2. No direct `model_validate` on worker outputs (must go through the worker
function that invokes guardrails):**

```bash
rg -nP 'TenKOutput\.model_validate|TranscriptOutput\.model_validate|EightKOutput\.model_validate|SFilingOutput\.model_validate' \
   src/auto_research/ \
   | rg -v 'tests/'
# Expected: zero hits outside tests. Workers construct via _call_llm + validate.
```

**3. No `permissive` / `soft_mode` / `skip_validation` flags on workers:**

```bash
rg -nP '(permissive|soft_mode|skip_validation|disable_guardrails|strict\s*=\s*False)' \
   src/auto_research/extract/
# Expected: zero hits.
```

**4. Quarantine path is wired and reachable:**

```bash
rg -nP 'DATA_QUARANTINE|data/quarantine' src/auto_research/extract/
# Expected: at least one write site per worker. If a worker has no
# quarantine write path, its CitationMismatch handler is incomplete.
```

**5. Property test exists for the worker you touched:**

```bash
# Replace ten_k with whichever worker you modified
rg -l 'corrupted_citation|CitationMismatch' tests/extract/test_ten_k_extraction.py
# Expected: one or more matches. If the test file doesn't exist or doesn't
# reference the mismatch path, add it before claiming done.
```

## Pre-submit checklist

- [ ] Every worker call routes through a wrapper that invokes
      `validate_citation_grounding`.
- [ ] Every `CitationMismatch` handler writes a `QuarantineRecord` and
      returns `None` (or re-raises).
- [ ] No `except` clause around guardrails just logs and continues.
- [ ] No bypass via `Output.model_validate(...)` in production code.
- [ ] No `permissive` / `soft_mode` / `skip_validation` arg introduced.
- [ ] No heuristic "repair" of mismatched citations.
- [ ] Property test feeding a corrupted citation and asserting quarantine
      write exists for the worker.
- [ ] DeepEval `hallucination` metric pre/post baselines captured (use
      `eval-baseline` skill when it lands; for now record by hand in the PR).
- [ ] PR body cites the test name and the DeepEval hallucination-rate delta.

## Escalation

If a swallow-and-continue pattern has already landed on `main`:

1. Treat as P0. Stop further extraction or eval work.
2. Identify all extraction runs since the merge by MLflow run timestamp.
3. Re-run the affected workers on the affected docs with the fixed validator.
4. Mark any signal performance result that consumed the silently-degraded
   outputs as invalid:
   `mlflow.set_tag("invalid_due_to", "citation_swallow_pr_#N")`.
5. Audit `data/quarantine/` — if it's been bulk-flushed or auto-cleaned,
   the audit trail is gone and the affected window must be re-extracted from
   `data/raw/` entirely.
6. Do not delete or rewrite `data/raw/` or `data/extracted/` during recovery.
   The `.claude/settings.json` deny rule will block destructive ops on those
   paths; if you find yourself wanting to bypass it, the fix is wrong.

## Related invariants

- INV-1 (PIT lag): if extraction is silently degraded, the Feast features it
  populates are also wrong. See [[pit-check]].
- INV-6 (prompt versioning): if you fixed a citation-grounding bug by editing
  a prompt, the prompt version must bump. See [[bump-prompt-version]].
