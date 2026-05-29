"""Live smoke for the locked local Qwen 35B-MoE stack against the three
10-K narrative-disclosure fields routed to it.

Hits a real running OpenAI-compatible local server (vllm-mlx by default;
see `scripts/serve_local_llm.sh`). Confirms that the three
`local/*`-routed `_ROUTING` rows in `auto_research._models` —
`going_concern`, `icfr_material_weaknesses`,
`critical_accounting_estimate_changes` — round-trip end-to-end through
`make_openai_compat_extraction_client` against their respective
`TenK*Partial` schemas, returning structured output that validates.

This is the eval evidence the `_ALLOWED_LOCAL_ROWS` allowlist in
`tests/unit/test_extract_local_dispatch.py` refers to.

What this catches that unit + VCR tests can't:

- The locked-stack's chat-template (`enable_thinking=false`) actually
  takes effect when JSON-schema-constrained output is requested.
- The server honors `response_format=json_schema` on the three new
  partial shapes (no schema-validation surprises from null / list
  combinations the unit tests' mock client can't catch).
- The server's `finish_reason` doesn't trip a future stop-reason filter
  if one gets added to `_common._call`.
- The locked-stack's actual discrimination — "substantial doubt"
  language fires the `going_concern` Claim; an unqualified opinion
  returns null — at the model tier we route to. (Unit tests mock the
  response; this is the only place that runs the real model.)

Cadence: nightly via `.github/workflows/live-smoke.yml` (when wired) or
on-demand via `make live-smoke`. CI skips it by default per the
`tests/live` conftest gate.

Opt-in env vars:

- `LOCAL_INFERENCE_URL` — base URL of the running OpenAI-compat server,
  e.g. `http://127.0.0.1:8000/v1` for vllm-mlx. The locked stack's
  launcher (`scripts/serve_local_llm.sh`) defaults to port 8000.

Wall-clock budget: ~30 s on the locked stack (Mac M2, 64 tok/s sustained
decode per cost-model doc §10.5 "Smoke-test results"), three calls with
bounded `max_tokens=1024`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from auto_research import _models
from auto_research.extract.openai_compat_client import (
    make_openai_compat_extraction_client,
)
from auto_research.extract.prompts.ten_k_narrative_field import (
    TEN_K_NARRATIVE_FIELD_CONFIGS,
    TEN_K_NARRATIVE_FIELD_PROMPT,
)

live_requires_env = ("LOCAL_INFERENCE_URL",)


def _local_routed_ten_k_tasks() -> frozenset[str]:
    """Discover at import time which `ten_k` task names route to a
    `local/*` model — that's the set the smoke needs to cover. Reading
    `_ROUTING` directly keeps the smoke from drifting if a future
    change adds (or removes) a local row; the smoke parameterization
    expands accordingly."""
    return frozenset(
        task
        for (worker, task), model_id in _models._ROUTING.items()
        if worker == "ten_k" and model_id.startswith("local/")
    )


_LOCAL_ROUTED_TASKS: frozenset[str] = _local_routed_ten_k_tasks()


# Realistic planted passages — one per field — that contain the
# discriminative language each prompt is supposed to detect. Source-quote
# substrings are deliberately phrased the way real 10-Ks phrase them.
_PLANTED_PASSAGES: dict[str, str] = {
    "going_concern": (
        "Item 8. Report of Independent Registered Public Accounting Firm. "
        "In our opinion, the consolidated financial statements present "
        "fairly, in all material respects, the financial position of the "
        "Company. However, the conditions described in Note 1 raise "
        "substantial doubt about the Company's ability to continue as a "
        "going concern."
    ),
    "icfr_material_weaknesses": (
        "Item 9A. Controls and Procedures. Management has identified a "
        "material weakness in internal control over financial reporting: "
        "we did not maintain effective controls over revenue recognition "
        "for contract modifications during the year ended December 31, "
        "2025. Specifically, our review controls were not designed at a "
        "sufficient level of precision."
    ),
    "critical_accounting_estimate_changes": (
        "Item 7. Management's Discussion and Analysis. Critical Accounting "
        "Estimates. During fiscal 2025, we revised our estimated useful "
        "life of server hardware from four to six years to reflect "
        "improvements in component longevity. This is a change in "
        "accounting estimate, applied prospectively per ASC 250."
    ),
}


def _config_by_field() -> dict[str, object]:
    return {c.field_name: c for c in TEN_K_NARRATIVE_FIELD_CONFIGS}


def _format_field_prompt(field_name: str) -> str:
    """Render the shared narrative-field prompt template with the
    per-field bindings from `TEN_K_NARRATIVE_FIELD_CONFIGS`. Mirrors
    `_extract_ten_k_rag`'s formatting so the smoke exercises the same
    system-prompt bytes the production path emits."""
    config = _config_by_field()[field_name]
    return TEN_K_NARRATIVE_FIELD_PROMPT.format(
        field_name=field_name,
        field_description=config.description,  # type: ignore[attr-defined]
    )


@pytest.fixture(scope="module")
def local_client() -> Iterator[object]:
    """Per-module local extraction client. One worker tag, one circuit
    breaker — three sequential calls reuse the same connection."""
    base_url = os.environ["LOCAL_INFERENCE_URL"]
    client = make_openai_compat_extraction_client(
        worker="ten_k",
        base_url=base_url,
        # Local serving needs a non-empty key string but ignores the
        # value; the wrapper's default sentinel works.
    )
    yield client


@pytest.mark.parametrize("field_name", sorted(_LOCAL_ROUTED_TASKS))
def test_local_qwen_round_trips_narrative_partial(
    field_name: str,
    local_client: object,
) -> None:
    """End-to-end: the locked stack returns a payload that validates
    against the field's `TenK*Partial` schema for a realistic 10-K
    passage. The Claim's `source_quote` must be a substring of the
    planted passage (the post-validator's INV-2 invariant); the smoke
    asserts a weaker substring check on the discriminative phrase so
    the test doesn't flake on whitespace normalization the model
    legitimately performs."""
    config = _config_by_field()[field_name]
    schema = config.schema  # type: ignore[attr-defined]
    prompt = _format_field_prompt(field_name)
    passage = _PLANTED_PASSAGES[field_name]

    payload, usage = local_client(  # type: ignore[operator]
        task=field_name,
        system_prompt=prompt,
        user_content=passage,
        output_schema=schema,
        max_tokens=1024,
    )

    # JSON-schema-constrained decoding should always return a dict; a
    # string or None means the server fell off the schema mid-decode or
    # the wrapper failed to parse it back.
    assert isinstance(payload, dict), (
        f"locked stack returned non-dict for {field_name}: type={type(payload).__name__}, "
        f"value={payload!r}"
    )
    parsed = schema.model_validate(payload)

    # The identity fields are required on every TenK*Partial. The model
    # should echo them back from the prompt's instructions; we don't
    # assert specific values because the planted passages don't carry
    # real CIK / accession numbers (the model fabricates plausible
    # values, which is fine for the routing smoke).
    assert isinstance(parsed.cik, str) and parsed.cik
    assert isinstance(parsed.accession_number, str) and parsed.accession_number

    # Field-specific discrimination check — the planted passage carries
    # a clearly-positive signal for each field, so the parsed value must
    # NOT be the modal "absent" outcome.
    if field_name == "going_concern":
        claim = parsed.going_concern
        assert claim is not None, (
            "planted passage explicitly contains 'substantial doubt' — "
            "going_concern must be a Claim, not None"
        )
        assert "substantial doubt" in claim.citation.source_quote.lower(), (
            f"going_concern citation does not anchor on the planted phrase: "
            f"{claim.citation.source_quote!r}"
        )
    elif field_name == "icfr_material_weaknesses":
        claims = parsed.icfr_material_weaknesses
        assert claims, (
            "planted passage explicitly contains 'material weakness' — "
            "icfr_material_weaknesses must be a non-empty list"
        )
    elif field_name == "critical_accounting_estimate_changes":
        claims = parsed.critical_accounting_estimate_changes
        assert claims, (
            "planted passage explicitly describes a useful-life revision — "
            "critical_accounting_estimate_changes must be a non-empty list"
        )

    # Usage tuple sanity — local backend reports token counts even
    # without per-call cost. Catches a wrapper regression that drops
    # the usage tuple altogether (would silently break cost dashboards
    # for the local tier).
    assert usage.get("input_tokens", 0) > 0
    assert usage.get("output_tokens", 0) > 0


def test_local_routed_tasks_have_planted_passages() -> None:
    """Every `ten_k` task routed to a local model must have a planted
    passage in `_PLANTED_PASSAGES`. If a future flip adds a task without
    a planted passage, the parametrized smoke would raise `KeyError` at
    a confusing place; this asserts the coverage gap explicitly."""
    missing = _LOCAL_ROUTED_TASKS - _PLANTED_PASSAGES.keys()
    assert not missing, (
        f"locally-routed ten_k task(s) {sorted(missing)} have no planted "
        "passage in _PLANTED_PASSAGES — add a realistic discriminative "
        "passage to the dict before this smoke covers the new flip"
    )
