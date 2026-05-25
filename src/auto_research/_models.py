"""Tiered model routing per spec §7.3.

`route_model(worker, task) -> str` returns the Anthropic model ID for a
given extraction worker or agent task. Lives at the package root (not
under `extract/`) because the routing table covers extraction workers
*and* the LangGraph research agent (§11) *and* the live critic (§12).
Putting it under `extract/` would force `agents/` to import from
`extract/` — a layering inversion.

**Naming convention.** For extraction workers, `task` is the literal
name of the output-model field whose extraction this call serves —
e.g., `("ten_k", "supplier_mentions")` matches
`TenKOutput.supplier_mentions`. This means a worker calling
`route_model(self.worker_name, output_field_name)` always resolves, and
schema additions in `extract/schemas.py` map 1:1 to new routing rows.
For agent / critic tasks that don't correspond to a specific output
field, use `"default"` and `"hard_critique"`.

The table is intentionally small and explicit. Each row is justified by
the spec's reasoning column (cross-doc reasoning, pattern recognition,
hard critique, …). Adding a row requires reading §7.3 and updating both
the table and the test; that friction is the point — silent
"sonnet-by-default" routing is exactly the bug we're preventing.

Unknown `(worker, task)` raises `ValueError`, not a silent fallback,
because:

- A typo in a call site (`"supplier_mentionn"`) would otherwise downgrade
  to a default model with no warning, producing quality regressions
  that show up only in eval suites.
- Cost attribution depends on knowing which model produced each output;
  a silent default smears that.

Model IDs are the un-dated aliases (e.g., `claude-sonnet-4-6`, not
`claude-sonnet-4-6-20251001`). The Anthropic API accepts both; the alias
form is stable across the small auto-rev bumps that happen between
major releases.
"""

from __future__ import annotations

from typing import Final

_SONNET: Final = "claude-sonnet-4-6"
_HAIKU: Final = "claude-haiku-4-5"
_OPUS: Final = "claude-opus-4-7"

# (worker, task) → model id. Sourced from `docs/specs/2026-05-22-design.md`
# §7.3. `task` keys match the schema field names in `extract/schemas.py`
# verbatim (e.g., `supplier_mentions`, not `supplier_mapping`), so a
# worker pulling fields by name routes consistently. New rows: add here,
# add a test in `tests/unit/test_models.py`, point at the spec line in
# the PR body.
_ROUTING: Final[dict[tuple[str, str], str]] = {
    # 10-K (TenKOutput): cross-doc reasoning ⇒ Sonnet; templated ⇒ Haiku.
    ("ten_k", "supplier_mentions"): _SONNET,
    ("ten_k", "customer_mentions"): _SONNET,
    ("ten_k", "guidance_tone"): _HAIKU,
    ("ten_k", "accrual_flags"): _HAIKU,
    ("ten_k", "risk_factor_deltas"): _HAIKU,
    # Transcripts (TranscriptOutput): Q&A nuance ⇒ Sonnet; prepared remarks ⇒ Haiku.
    ("transcript", "q_and_a_evasiveness"): _SONNET,
    ("transcript", "prepared_remarks_tone"): _HAIKU,
    ("transcript", "forward_statements"): _SONNET,
    # 8-K (EightKOutput) + S-filings (SFilingOutput): high-volume pattern recognition ⇒ Haiku.
    ("eight_k", "event_classification"): _HAIKU,
    ("eight_k", "milestone_mentions"): _HAIKU,
    ("eight_k", "dilution_language_flags"): _HAIKU,
    ("s_filings", "dilution_event"): _HAIKU,
    ("s_filings", "capital_raise_language"): _HAIKU,
    ("s_filings", "use_of_proceeds"): _HAIKU,
    # Contextual chunking (Issue #14): one-line "this chunk is from X" rewrite
    # per ChildChunk for Anthropic's contextual-retrieval pattern. High-volume
    # templated rewrite ⇒ Haiku per §7.3.
    ("extract", "contextual_chunk"): _HAIKU,
    # Agents don't have output schemas; use logical task names instead of fields.
    ("research_agent", "default"): _SONNET,
    ("research_agent", "hard_critique"): _OPUS,
    ("live_critic", "default"): _SONNET,
    ("live_critic", "hard_critique"): _OPUS,
}


def route_model(worker: str, task: str) -> str:
    """Return the Anthropic model ID for `(worker, task)`.

    Raises `ValueError` (not KeyError) with the offending pair in the
    message — callers should not catch this; a missing entry is a code
    bug that wants a route-table edit + spec justification, not a
    runtime fallback.
    """
    model = _ROUTING.get((worker, task))
    if model is None:
        raise ValueError(
            f"no model routed for (worker={worker!r}, task={task!r}); "
            "add a row to `_ROUTING` in `auto_research/_models.py` with "
            "justification from `docs/specs/2026-05-22-design.md` §7.3"
        )
    return model


__all__ = ["route_model"]
