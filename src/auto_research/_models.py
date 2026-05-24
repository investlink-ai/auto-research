"""Tiered model routing per spec §7.3.

`route_model(worker, task) -> str` returns the Anthropic model ID for a
given extraction worker or agent task. Lives at the package root (not
under `extract/`) because the routing table covers extraction workers
*and* the LangGraph research agent (§11) *and* the live critic (§12).
Putting it under `extract/` would force `agents/` to import from
`extract/` — a layering inversion.

The table is intentionally small and explicit. Each row is justified by
the spec's reasoning column (cross-doc reasoning, pattern recognition,
hard critique, …). Adding a row requires reading §7.3 and updating both
the table and the test; that friction is the point — silent
"sonnet-by-default" routing is exactly the bug we're preventing.

Unknown `(worker, task)` raises `ValueError`, not a silent fallback,
because:

- A typo in a call site (`"supplier_mappin"`) would otherwise downgrade
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
# §7.3. New rows: add here, add a test in `tests/unit/test_models.py`,
# point at the spec line in the PR body.
_ROUTING: Final[dict[tuple[str, str], str]] = {
    # 10-K: cross-doc reasoning ⇒ Sonnet; templated fields ⇒ Haiku.
    ("ten_k", "supplier_mapping"): _SONNET,
    ("ten_k", "customer_mapping"): _SONNET,
    ("ten_k", "guidance_tone"): _HAIKU,
    ("ten_k", "accrual_flags"): _HAIKU,
    ("ten_k", "risk_factor_deltas"): _HAIKU,
    # Transcripts: Q&A nuance ⇒ Sonnet; prepared-remarks tone ⇒ Haiku.
    ("transcript", "evasiveness"): _SONNET,
    ("transcript", "remarks_tone"): _HAIKU,
    ("transcript", "forward_statements"): _SONNET,
    # 8-K + S-filings: high-volume pattern recognition ⇒ Haiku.
    ("eight_k", "event_classification"): _HAIKU,
    ("eight_k", "milestone_mentions"): _HAIKU,
    ("eight_k", "dilution_language"): _HAIKU,
    ("s_filings", "dilution_language"): _HAIKU,
    ("s_filings", "capital_raise_language"): _HAIKU,
    ("s_filings", "use_of_proceeds"): _HAIKU,
    # Research agent: Sonnet default, Opus for hard critiques only.
    ("research_agent", "default"): _SONNET,
    ("research_agent", "hard_critique"): _OPUS,
    # Live critic: same pattern.
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
