# Issue #11 — Prompts Registry + S-1/S-3 Worker + Lifecycle Discipline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the `extract/prompts/` convention and the first end-to-end extraction worker (S-1/S-3 dilution language + form classification), with prompt-lifecycle discipline mechanically enforced — cache key captures the full completion config, prompt and output schema co-versioned, promotion to the Langfuse `production` tag gated by an eval threshold.

**Architecture:**
- Prompts live in code (one file per prompt, `*_PROMPT_VERSION` constant colocated). Langfuse is the *registry* — pushed from code, not the source of truth at runtime. Workers read the in-code prompt; Langfuse holds the canonical version history + tag state for audit/replay/promotion.
- Content-hash cache keyed on `sha256(raw_doc_bytes + prompt_version + schema_version + model_id + decoding_params_json)`. File-backed JSON under `data/cache/extract/<worker>/<sha>.json`. Hit → load → return; miss → call LLM → write → return. INV-6 tightened: completion-config-pinned, not just prompt-pinned.
- `SCHEMA_VERSION` lives as a `ClassVar` on each Pydantic output model. Included in cache key. Not a field, so JSON dumps stay stable.
- Promotion to Langfuse `production` tag is gated by `scripts/promote_prompt.py` — runs a gold-set F1 evaluation, refuses to flip if F1 < threshold or token cost > ceiling.
- `bump-prompt-version` skill extended with a schema-co-versioning check: if `extract/schemas.py` field shape changes in a diff but no `*_PROMPT_VERSION` constant bumped, the precommit check fails.

**Tech Stack:** Python 3.12, `anthropic`, `pydantic>=2.13`, `langfuse>=2,<3`, `pytest`, `vcrpy` (for the integration test), `mypy`, `ruff`.

**Tier classification (per `docs/AI_WORKFLOW.md` §2):** **Tier 2** — touches `extract/schemas.py` (sensitive). PR body must include the Change Contract block + sensitive-path evidence table.

---

## File Structure

### New files
| Path | Responsibility |
|---|---|
| `src/auto_research/extract/prompts/__init__.py` | Package marker |
| `src/auto_research/extract/prompts/s_filings_dilution.py` | The S-1/S-3 dilution-language prompt + `S_FILINGS_DILUTION_PROMPT_VERSION = "v1"` constant |
| `src/auto_research/extract/prompts/_registry.py` | Thin Langfuse wrapper: `register_prompt(name, version, text)`, `set_prompt_tag(name, version, tag)` |
| `src/auto_research/extract/cache.py` | Content-hash idempotent cache: `cache_key(...)`, `read(...)`, `write(...)` |
| `src/auto_research/extract/workers/__init__.py` | Package marker |
| `src/auto_research/extract/workers/s_filings.py` | The S-1/S-3 worker: orchestrates prompt → client → cache → guardrails → output |
| `tests/unit/test_extract_cache.py` | Cache-key + read/write tests |
| `tests/unit/test_extract_prompts_registry.py` | Langfuse wrapper tests (mocked SDK) |
| `tests/unit/test_extract_worker_s_filings.py` | Worker tests with mocked Anthropic client |
| `tests/integration/test_extract_worker_s_filings_vcr.py` | One end-to-end VCR-replayed test against a real S-3 fixture |
| `tests/fixtures/s_filings/sample_s3_excerpt.txt` | ~5-10KB hand-trimmed S-3 (cover + dilution section) |
| `tests/fixtures/s_filings/sample_s3_expected.json` | Hand-labeled `SFilingOutput` for the fixture |
| `tests/fixtures/s_filings/corrupted_citation_output.json` | LLM output JSON with a deliberately wrong `source_span` |
| `eval/baselines/s_filings_dilution__gold.json` | Minimal hand-labeled gold set (2-3 samples) |
| `scripts/promote_prompt.py` | Eval-gated tag-flip script |
| `tests/unit/test_promote_prompt.py` | Promotion-script unit tests |

### Modified files
| Path | Change |
|---|---|
| `src/auto_research/extract/schemas.py` | Add `SCHEMA_VERSION: ClassVar[str] = "v1"` to each output model |
| `src/auto_research/extract/__init__.py` | Re-export the new worker entry point |
| `AGENTS.md` | Update INV-6 wording: `(raw_doc, prompt_version)` → `(raw_doc, prompt_version, schema_version, model_id, decoding_params)` |
| `docs/CONTRACTS.md` | Same wording update at lines 90-92 |
| `.gitignore` | Add `data/cache/`, `data/quarantine/` (if not already) |
| `.claude/skills/bump-prompt-version/SKILL.md` | Add check #6: schema-co-versioning |
| `README.md` | Add a "Prompt lifecycle" section pointing at the workflow |

### Decisions worth flagging up front
- **Code is the source of truth for prompts; Langfuse is the registry.** Workers read the in-code constant. Langfuse holds version history + tag state. This avoids "Langfuse down → no extraction" and keeps the cache key deterministic without a network round-trip.
- **`SCHEMA_VERSION` is a `ClassVar`, not a Pydantic field.** Keeps JSON dump shape stable; cache key reads it via `SFilingOutput.SCHEMA_VERSION`.
- **Cache lives at `data/cache/extract/<worker>/<sha>.json`.** Pure-hash filenames (debuggable via cat of any file — full metadata stored in the cache record).
- **`decoding_params` for v1 = `{"max_tokens": 4096, "temperature": null}`** (the SDK default — we don't set temperature, Anthropic uses 1.0). Frozen as a dict for the cache key; future-proofs against per-worker overrides.
- **Gold set v1 is small (2-3 examples).** Hand-labeling 20/worker is W2+ work; the *script* and *contract* land here, the gold-set growth is incremental.
- **DeepEval is NOT added in this issue.** `promote_prompt.py` uses a hand-rolled F1 over expected fields. Trivially swappable later. Avoids dragging a heavy dep into a Tier-2 issue.

---

## Task list

### Task 1: Prompts registry scaffold

**Files:**
- Create: `src/auto_research/extract/prompts/__init__.py`
- Create: `src/auto_research/extract/prompts/s_filings_dilution.py`
- Test: `tests/unit/test_extract_prompts.py`

- [ ] **Step 1.1: Write the failing test**

```python
# tests/unit/test_extract_prompts.py
"""Unit tests for the prompts registry convention (Issue #11)."""

from __future__ import annotations

import re

from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT,
    S_FILINGS_DILUTION_PROMPT_VERSION,
)


def test_prompt_version_is_human_readable_tag() -> None:
    """Versions must be `vN` form, not hashes (see bump-prompt-version skill)."""
    assert re.fullmatch(r"v\d+", S_FILINGS_DILUTION_PROMPT_VERSION)


def test_prompt_text_carries_required_extraction_contract() -> None:
    """Every extraction prompt must instruct the model to emit `source_span`
    and `source_quote` for every claim — that's the INV-2 wire format."""
    assert "source_span" in S_FILINGS_DILUTION_PROMPT
    assert "source_quote" in S_FILINGS_DILUTION_PROMPT


def test_prompt_text_carries_placeholder() -> None:
    """Prompt is a template with a `{source_text}` placeholder."""
    assert "{source_text}" in S_FILINGS_DILUTION_PROMPT
```

- [ ] **Step 1.2: Run to confirm failure**

```bash
cd ~/Documents/projects/auto-research/.worktree/11-prompts-registry-s1s3
uv run pytest tests/unit/test_extract_prompts.py -v
# Expected: ImportError / ModuleNotFoundError on prompts.s_filings_dilution
```

- [ ] **Step 1.3: Create package + prompt file**

```python
# src/auto_research/extract/prompts/__init__.py
"""Per-prompt modules — one file per prompt, version constant colocated.

Convention (defended by the `bump-prompt-version` skill):

- File name = prompt name (`s_filings_dilution.py`, `ten_k_guidance.py`, ...).
- Module exports two constants: `<NAME>_PROMPT_VERSION` (a `vN` tag) and
  `<NAME>_PROMPT` (the template text).
- Code is the source of truth at runtime. Langfuse is the registry —
  pushed from code via `_registry.register_prompt` for version-history
  and tag state (`dev` / `staging` / `production`).

A prompt edit that doesn't bump `*_PROMPT_VERSION` silently corrupts the
content-hash cache. The `bump-prompt-version` skill is the mechanical
guard; this docstring is the convention reference.
"""
```

```python
# src/auto_research/extract/prompts/s_filings_dilution.py
"""S-1 / S-3 dilution-language extraction prompt (Issue #11)."""

from __future__ import annotations

S_FILINGS_DILUTION_PROMPT_VERSION = "v1"

S_FILINGS_DILUTION_PROMPT = """\
You are extracting structured dilution and capital-raise signals from an SEC
S-1 or S-3 registration statement.

Read <source_text> carefully and return a single JSON object matching the
SFilingOutput schema. Every claim MUST include:
- source_span: tuple [start_char, end_char] giving byte offsets into
  <source_text>.
- source_quote: the verbatim slice source_text[start_char:end_char]. If the
  quote does not appear verbatim in source_text at exactly that span, the
  output will be rejected.

Fields to populate:
- cik: the issuer's CIK (10-digit, leading zeros).
- accession_number: the filing's SEC accession number.
- form_type: "S-1" or "S-3".
- dilution_event: a single Claim describing the headline dilution event
  (e.g., "shelf takedown of $200M common stock"), with confidence in [0, 1].
- capital_raise_language: list of Claims for each distinct capital-raise
  phrase in the filing (e.g., "at-the-market offering", "registered direct").
- use_of_proceeds: list of Claims describing intended uses (e.g.,
  "general corporate purposes", "fund Phase II clinical trial").

Do not invent quotes. If a field has no support in source_text, return an
empty list rather than fabricating a citation.

<source_text>
{source_text}
</source_text>
"""

__all__ = [
    "S_FILINGS_DILUTION_PROMPT",
    "S_FILINGS_DILUTION_PROMPT_VERSION",
]
```

- [ ] **Step 1.4: Run tests, expect green**

```bash
uv run pytest tests/unit/test_extract_prompts.py -v
# Expected: 3 passed
```

- [ ] **Step 1.5: Commit**

```bash
git add src/auto_research/extract/prompts/ tests/unit/test_extract_prompts.py
git commit -m "feat(extract): prompts registry convention + s_filings_dilution v1 (#11)"
```

---

### Task 2: `SCHEMA_VERSION` ClassVar on output models

**Files:**
- Modify: `src/auto_research/extract/schemas.py`
- Test: `tests/unit/test_extract_schemas.py` (add cases)

- [ ] **Step 2.1: Write failing tests**

Append to `tests/unit/test_extract_schemas.py`:

```python
from typing import ClassVar, get_type_hints

from auto_research.extract.schemas import (
    EightKOutput,
    SFilingOutput,
    TenKOutput,
    TranscriptOutput,
)


def test_every_output_model_carries_schema_version() -> None:
    """Every Pydantic output model exports `SCHEMA_VERSION` as a ClassVar.

    Cache key includes `schema_version`; if a model's field shape changes
    without bumping `SCHEMA_VERSION`, cached parquet rows deserialize wrong
    on next read. See AGENTS.md INV-6.
    """
    for cls in (SFilingOutput, TenKOutput, TranscriptOutput, EightKOutput):
        assert hasattr(cls, "SCHEMA_VERSION"), f"{cls.__name__} missing SCHEMA_VERSION"
        assert isinstance(cls.SCHEMA_VERSION, str)
        assert cls.SCHEMA_VERSION.startswith("v")


def test_schema_version_is_classvar_not_pydantic_field() -> None:
    """`SCHEMA_VERSION` must NOT appear in `model_fields` — it's metadata,
    not data. If it leaks into the Pydantic field set, dumps will include
    it and downstream consumers will break."""
    for cls in (SFilingOutput, TenKOutput, TranscriptOutput, EightKOutput):
        assert "SCHEMA_VERSION" not in cls.model_fields
```

- [ ] **Step 2.2: Run, confirm fail**

```bash
uv run pytest tests/unit/test_extract_schemas.py::test_every_output_model_carries_schema_version -v
# Expected: AttributeError: type object 'SFilingOutput' has no attribute 'SCHEMA_VERSION'
```

- [ ] **Step 2.3: Add `ClassVar` to each output model**

In `src/auto_research/extract/schemas.py`:

```python
# At the imports — add ClassVar
from typing import Annotated, ClassVar
```

Then in each of the four output models (`TenKOutput`, `TranscriptOutput`, `EightKOutput`, `SFilingOutput`), immediately under `model_config = _FROZEN_STRICT`, add:

```python
    SCHEMA_VERSION: ClassVar[str] = "v1"
```

- [ ] **Step 2.4: Run, expect green**

```bash
uv run pytest tests/unit/test_extract_schemas.py -v
# Expected: all green, including the two new tests
```

- [ ] **Step 2.5: Commit**

```bash
git add src/auto_research/extract/schemas.py tests/unit/test_extract_schemas.py
git commit -m "feat(extract): SCHEMA_VERSION ClassVar on output models (#11)"
```

---

### Task 3: Content-hash cache module

**Files:**
- Create: `src/auto_research/extract/cache.py`
- Test: `tests/unit/test_extract_cache.py`

- [ ] **Step 3.1: Write failing tests**

```python
# tests/unit/test_extract_cache.py
"""Unit tests for the content-hash idempotent cache (Issue #11).

Defends INV-6: cache key captures the full completion config — raw_doc,
prompt_version, schema_version, model_id, decoding_params. A change to ANY
of the five must produce a fresh cache key (and thus a fresh LLM call).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_research.extract.cache import cache_key, read, write


def test_cache_key_is_stable_for_same_inputs() -> None:
    k1 = cache_key(
        raw_doc=b"hello",
        prompt_version="v1",
        schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"max_tokens": 4096},
    )
    k2 = cache_key(
        raw_doc=b"hello",
        prompt_version="v1",
        schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"max_tokens": 4096},
    )
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


@pytest.mark.parametrize(
    "field",
    ["raw_doc", "prompt_version", "schema_version", "model_id", "decoding_params"],
)
def test_cache_key_changes_when_any_completion_config_input_changes(field: str) -> None:
    """The interview-grade test: tiered routing (Haiku→Sonnet swap) must
    not silently reuse stale cache. Each of the five inputs is a cache-key
    component; flip any one and the key changes."""
    base = dict(
        raw_doc=b"hello",
        prompt_version="v1",
        schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"max_tokens": 4096},
    )
    mutated = {
        "raw_doc": b"hello world",
        "prompt_version": "v2",
        "schema_version": "v2",
        "model_id": "claude-sonnet-4-6",
        "decoding_params": {"max_tokens": 8192},
    }
    other = dict(base) | {field: mutated[field]}
    assert cache_key(**base) != cache_key(**other), f"changing {field} must change the cache key"


def test_decoding_params_dict_ordering_does_not_affect_key() -> None:
    """Two dicts with the same items in different insertion order must hash
    to the same key (canonical JSON serialization)."""
    k_ab = cache_key(
        raw_doc=b"x", prompt_version="v1", schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"a": 1, "b": 2},
    )
    k_ba = cache_key(
        raw_doc=b"x", prompt_version="v1", schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"b": 2, "a": 1},
    )
    assert k_ab == k_ba


def test_round_trip_write_then_read(tmp_path: Path) -> None:
    key = cache_key(
        raw_doc=b"x", prompt_version="v1", schema_version="v1",
        model_id="claude-haiku-4-5", decoding_params={},
    )
    payload = {"hello": "world", "n": 42}
    write(tmp_path, "s_filings", key, payload)
    got = read(tmp_path, "s_filings", key)
    assert got == payload


def test_read_returns_none_on_miss(tmp_path: Path) -> None:
    assert read(tmp_path, "s_filings", "deadbeef" * 8) is None
```

- [ ] **Step 3.2: Run, confirm fail**

```bash
uv run pytest tests/unit/test_extract_cache.py -v
# Expected: ModuleNotFoundError on auto_research.extract.cache
```

- [ ] **Step 3.3: Implement the cache module**

```python
# src/auto_research/extract/cache.py
"""Content-hash idempotent cache for extraction worker outputs (Issue #11).

Defends INV-6: extraction is a pure function of the full completion config,
not just the prompt version. The cache key is

    sha256(raw_doc_bytes + prompt_version + schema_version + model_id +
           canonical_json(decoding_params))

so changing the routed model, the decoding temperature, the prompt text
(via its version), or the output schema shape (via its version) all
produce a fresh key and force a fresh LLM call. The original
`(raw_doc, prompt_version)` formulation was wrong: tiered routing
(Haiku→Sonnet swap) would silently reuse stale cache. The lifecycle
discussion that motivated this is captured in Issue #11.

Storage: one JSON file per cache entry at
`<root>/<worker>/<sha>.json`. Pure-hash filenames — the full
provenance metadata is inside the file so any single `cat <sha>.json`
shows what produced it. Atomic writes via `auto_research._io`.

Not in this module:
- LRU / size-based eviction. The cache is content-addressed; growth is
  bounded by `len(raw_docs) × len(distinct_completion_configs)`. For
  the four extraction workers across ~2,700 docs that's ~12K files
  steady-state — well under any reasonable disk budget.
- Async I/O. Workers are nightly batch; the marginal latency from a
  sync read is irrelevant against the ~seconds-per-LLM-call regime.
- A cache invalidation API. Bumping `*_PROMPT_VERSION` or `SCHEMA_VERSION`
  is the invalidation primitive — old keys simply become unreferenced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from auto_research._io import atomic_write_text


def _canonical_json(value: Any) -> str:
    """Stable JSON serialization for hash inputs: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def cache_key(
    *,
    raw_doc: bytes,
    prompt_version: str,
    schema_version: str,
    model_id: str,
    decoding_params: dict[str, Any],
) -> str:
    """Compute the sha256 cache key for a completion config.

    Returns a 64-char lowercase hex string.
    """
    h = hashlib.sha256()
    h.update(raw_doc)
    h.update(b"|")
    h.update(prompt_version.encode())
    h.update(b"|")
    h.update(schema_version.encode())
    h.update(b"|")
    h.update(model_id.encode())
    h.update(b"|")
    h.update(_canonical_json(decoding_params).encode())
    return h.hexdigest()


def read(root: Path, worker: str, key: str) -> dict[str, Any] | None:
    """Return the cached payload dict, or None on miss.

    `root` is the cache root (production: `data/cache/extract/`; tests:
    `tmp_path`). `worker` namespaces entries so a `find data/cache/extract/s_filings/`
    enumerates one worker's hits.
    """
    path = root / worker / f"{key}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record["payload"]  # type: ignore[no-any-return]


def write(root: Path, worker: str, key: str, payload: dict[str, Any]) -> None:
    """Persist `payload` keyed by `key`. Atomic — partial writes never leave
    a half-file behind on crash."""
    path = root / worker / f"{key}.json"
    record = {"key": key, "worker": worker, "payload": payload}
    atomic_write_text(path, json.dumps(record, indent=2))


DEFAULT_CACHE_ROOT = Path("data/cache/extract")

__all__ = ["DEFAULT_CACHE_ROOT", "cache_key", "read", "write"]
```

- [ ] **Step 3.4: Run, expect green**

```bash
uv run pytest tests/unit/test_extract_cache.py -v
# Expected: all green
```

- [ ] **Step 3.5: Commit**

```bash
git add src/auto_research/extract/cache.py tests/unit/test_extract_cache.py
git commit -m "feat(extract): content-hash cache keyed on full completion config (#11)"
```

---

### Task 4: Langfuse registry wrapper

**Files:**
- Create: `src/auto_research/extract/prompts/_registry.py`
- Test: `tests/unit/test_extract_prompts_registry.py`

- [ ] **Step 4.1: Write failing tests**

```python
# tests/unit/test_extract_prompts_registry.py
"""Unit tests for the Langfuse prompts-registry wrapper (Issue #11).

Workers don't fetch prompts from Langfuse at runtime — code is the source
of truth, this wrapper only pushes for visibility and flips tags for
promotion. We mock the Langfuse client so tests stay hermetic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from auto_research.extract.prompts._registry import (
    register_prompt,
    set_prompt_tag,
)


def test_register_prompt_calls_langfuse_create_prompt() -> None:
    client = MagicMock()
    register_prompt(
        name="s_filings_dilution",
        version="v1",
        text="extract dilution from {source_text}",
        client=client,
    )
    client.create_prompt.assert_called_once()
    kwargs = client.create_prompt.call_args.kwargs
    assert kwargs["name"] == "s_filings_dilution"
    assert kwargs["prompt"] == "extract dilution from {source_text}"
    # The Langfuse v2 SDK takes labels as the version-tag mechanism; we
    # include the code version constant as a label so a registry browser
    # can match Langfuse rows to code commits.
    assert "v1" in kwargs["labels"]


def test_set_prompt_tag_promotes_existing_version() -> None:
    client = MagicMock()
    set_prompt_tag(
        name="s_filings_dilution",
        version="v1",
        tag="production",
        client=client,
    )
    client.update_prompt.assert_called_once()
    kwargs = client.update_prompt.call_args.kwargs
    assert kwargs["name"] == "s_filings_dilution"
    assert "production" in kwargs["new_labels"]
```

- [ ] **Step 4.2: Run, confirm fail**

```bash
uv run pytest tests/unit/test_extract_prompts_registry.py -v
# Expected: ModuleNotFoundError
```

- [ ] **Step 4.3: Implement the wrapper**

```python
# src/auto_research/extract/prompts/_registry.py
"""Thin Langfuse wrapper for the prompt registry (Issue #11).

Two operations:

- `register_prompt(name, version, text)` — push a code-defined prompt
  into Langfuse so the registry has version history. Idempotent: pushing
  the same `(name, version, text)` twice is a no-op on the Langfuse side
  (it dedupes on prompt text + label set).
- `set_prompt_tag(name, version, tag)` — flip a Langfuse label
  (`dev` / `staging` / `production`) onto a specific version. Used by
  `scripts/promote_prompt.py` after a gold-set eval gate passes.

What this module deliberately doesn't do:

- Fetch prompts at runtime. Workers read the in-code `*_PROMPT` /
  `*_PROMPT_VERSION` constants. A network round-trip per extraction
  would make the cache key non-deterministic (Langfuse latency) and
  introduce a runtime dep on a service that's allowed to be down.
- Diff in-code prompt vs Langfuse-registered prompt. That's the job of
  CI or a manual `python -m ...prompts._registry_list` invocation; this
  module stays small.

The Langfuse v2 SDK is sync; W1 doesn't need async for one-shot
registration calls.
"""

from __future__ import annotations

from typing import Any, Protocol


class _LangfuseClient(Protocol):
    """Minimal slice of `langfuse.Langfuse` we depend on. Lets us pass a
    `MagicMock()` in tests without a runtime Langfuse instance."""

    def create_prompt(self, **kwargs: Any) -> Any: ...
    def update_prompt(self, **kwargs: Any) -> Any: ...


def register_prompt(
    *,
    name: str,
    version: str,
    text: str,
    client: _LangfuseClient,
) -> None:
    """Push a code-defined prompt into the Langfuse registry."""
    client.create_prompt(
        name=name,
        prompt=text,
        labels=[version],
        type="text",
    )


def set_prompt_tag(
    *,
    name: str,
    version: str,
    tag: str,
    client: _LangfuseClient,
) -> None:
    """Flip a tag label (`dev`/`staging`/`production`) onto a version."""
    client.update_prompt(
        name=name,
        version=version,
        new_labels=[tag],
    )


__all__ = ["register_prompt", "set_prompt_tag"]
```

- [ ] **Step 4.4: Run, expect green**

```bash
uv run pytest tests/unit/test_extract_prompts_registry.py -v
```

- [ ] **Step 4.5: Commit**

```bash
git add src/auto_research/extract/prompts/_registry.py tests/unit/test_extract_prompts_registry.py
git commit -m "feat(extract): Langfuse prompt-registry wrapper (#11)"
```

---

### Task 5: S-1/S-3 worker (with mocked Anthropic)

**Files:**
- Create: `src/auto_research/extract/workers/__init__.py`
- Create: `src/auto_research/extract/workers/s_filings.py`
- Test: `tests/unit/test_extract_worker_s_filings.py`

- [ ] **Step 5.1: Write failing tests**

```python
# tests/unit/test_extract_worker_s_filings.py
"""Unit tests for the S-1/S-3 extraction worker (Issue #11).

End-to-end of `extract_s_filing`: prompt → Anthropic → JSON parse →
SFilingOutput → citation grounding → cache. We mock the Anthropic SDK
to make the test hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage

from auto_research.extract.workers.s_filings import extract_s_filing


_SAMPLE_S3 = (
    "This shelf takedown of $200 million of common stock will be used for "
    "general corporate purposes and to fund the Phase II clinical trial."
)


def _make_response(body: dict[str, Any]) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(body), citations=None)],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=100, output_tokens=50, cache_creation=None,
            cache_creation_input_tokens=None, cache_read_input_tokens=None,
            inference_geo=None, server_tool_use=None, service_tier="standard",
        ),
    )


def _valid_output_for(text: str) -> dict[str, Any]:
    # Build a valid SFilingOutput JSON whose citations point at real spans.
    quote = "shelf takedown of $200 million of common stock"
    start = text.find(quote)
    end = start + len(quote)
    return {
        "cik": "0000000001",
        "accession_number": "0000000001-25-000001",
        "form_type": "S-3",
        "dilution_event": {
            "citation": {"source_span": [start, end], "source_quote": quote},
            "confidence": 0.9,
        },
        "capital_raise_language": [],
        "use_of_proceeds": [],
    }


def _fake_client(body: dict[str, Any]) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_response(body)
    return cast(anthropic.Anthropic, fake)


def test_extract_s_filing_returns_validated_output(tmp_path: Path) -> None:
    client = _fake_client(_valid_output_for(_SAMPLE_S3))
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3,
        doc_id="test-001",
        cache_root=tmp_path,
        anthropic_client=client,
    )
    assert out is not None
    assert out.form_type == "S-3"
    assert out.dilution_event.confidence == pytest.approx(0.9)


def test_cache_hit_skips_llm_call(tmp_path: Path) -> None:
    """Second call with identical inputs must NOT touch the Anthropic SDK."""
    body = _valid_output_for(_SAMPLE_S3)
    client = _fake_client(body)
    first = extract_s_filing(
        raw_doc=_SAMPLE_S3, doc_id="test-001",
        cache_root=tmp_path, anthropic_client=client,
    )
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]
    second = extract_s_filing(
        raw_doc=_SAMPLE_S3, doc_id="test-001",
        cache_root=tmp_path, anthropic_client=client,
    )
    # No new SDK call:
    assert client.messages.create.call_count == 1  # type: ignore[attr-defined]
    assert first == second


def test_corrupted_citation_routes_to_quarantine(tmp_path: Path) -> None:
    bad = _valid_output_for(_SAMPLE_S3)
    # Move the span by 1 character — `source_quote` no longer matches the slice
    bad["dilution_event"]["citation"]["source_span"] = [
        bad["dilution_event"]["citation"]["source_span"][0] + 1,
        bad["dilution_event"]["citation"]["source_span"][1] + 1,
    ]
    client = _fake_client(bad)
    out = extract_s_filing(
        raw_doc=_SAMPLE_S3, doc_id="bad-001",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quarantine",
        anthropic_client=client,
    )
    assert out is None
    qfile = tmp_path / "quarantine" / "s_filings" / "bad-001.json"
    assert qfile.exists()
```

- [ ] **Step 5.2: Confirm tests fail**

```bash
uv run pytest tests/unit/test_extract_worker_s_filings.py -v
# Expected: ModuleNotFoundError
```

- [ ] **Step 5.3: Implement the worker**

```python
# src/auto_research/extract/workers/__init__.py
"""Per-worker extraction modules. Each module exposes one entry function
`extract_<worker>(raw_doc, doc_id, ...) -> Output | None`. None means
quarantined; the caller MUST NOT persist any part of a None result."""
```

```python
# src/auto_research/extract/workers/s_filings.py
"""S-1 / S-3 extraction worker — first end-to-end validator (Issue #11).

Composes the W1 extraction primitives:

    prompt + client + cache + guardrails → SFilingOutput | None

Flow:

1. Build the cache key from the full completion config (raw_doc bytes,
   prompt version, schema version, routed model, decoding params).
2. Look up `data/cache/extract/s_filings/<sha>.json`. Hit → deserialize
   into `SFilingOutput`, return.
3. Miss → invoke the Anthropic client (with reliability + caching from
   `make_extraction_client`), parse the JSON content block into
   `SFilingOutput`, validate via `validate_or_quarantine`.
4. On validation success: write to cache, return the output. On failure:
   the guardrail already wrote a QuarantineRecord; return None.

The function takes both `cache_root` and `quarantine_root` so tests can
pass `tmp_path` and stay hermetic. Production callers omit them and get
the package defaults.

`anthropic_client` is injected the same way `make_extraction_client`
accepts it — production callers omit it; tests pass a MagicMock.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from auto_research._models import route_model
from auto_research.extract import cache as content_cache
from auto_research.extract.client import make_extraction_client
from auto_research.extract.guardrails import (
    DEFAULT_QUARANTINE_ROOT,
    validate_or_quarantine,
)
from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT,
    S_FILINGS_DILUTION_PROMPT_VERSION,
)
from auto_research.extract.schemas import SFilingOutput

_WORKER = "s_filings"
_TASK = "dilution_event"  # matches SFilingOutput.dilution_event field name
_DECODING_PARAMS: dict[str, object] = {"max_tokens": 4096}


def extract_s_filing(
    *,
    raw_doc: str,
    doc_id: str,
    cache_root: Path | None = None,
    quarantine_root: Path | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
) -> SFilingOutput | None:
    """Extract an SFilingOutput from a raw S-1/S-3 text.

    Returns `None` when the output failed citation grounding; the caller
    MUST treat None as "do not persist."
    """
    cache_root = cache_root if cache_root is not None else content_cache.DEFAULT_CACHE_ROOT
    quarantine_root = (
        quarantine_root if quarantine_root is not None else DEFAULT_QUARANTINE_ROOT
    )
    model_id = route_model(_WORKER, _TASK)

    key = content_cache.cache_key(
        raw_doc=raw_doc.encode(),
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        schema_version=SFilingOutput.SCHEMA_VERSION,
        model_id=model_id,
        decoding_params=_DECODING_PARAMS,
    )

    cached = content_cache.read(cache_root, _WORKER, key)
    if cached is not None:
        return SFilingOutput.model_validate(cached)

    client = make_extraction_client(
        worker=_WORKER,
        anthropic_client=anthropic_client,
    )
    response = client(
        task=_TASK,
        system_prompt=S_FILINGS_DILUTION_PROMPT.format(source_text=raw_doc),
        user_content=raw_doc,
        max_tokens=_DECODING_PARAMS["max_tokens"],  # type: ignore[arg-type]
    )

    # Anthropic responses are a list of content blocks; the worker expects
    # one TextBlock containing JSON. Other shapes are a model misbehavior
    # surface — let `model_validate` raise so quarantine catches it.
    text = "".join(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    output = SFilingOutput.model_validate(parsed)

    validated = validate_or_quarantine(
        output,
        source_text=raw_doc,
        doc_id=doc_id,
        worker=_WORKER,
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        quarantine_root=quarantine_root,
    )
    if validated is None:
        return None

    content_cache.write(cache_root, _WORKER, key, validated.model_dump(mode="json"))
    return validated


__all__ = ["extract_s_filing"]
```

- [ ] **Step 5.4: Run tests, expect green**

```bash
uv run pytest tests/unit/test_extract_worker_s_filings.py -v
# Expected: 3 passed
```

- [ ] **Step 5.5: Commit**

```bash
git add src/auto_research/extract/workers/ tests/unit/test_extract_worker_s_filings.py
git commit -m "feat(extract): S-1/S-3 worker end-to-end (#11)"
```

---

### Task 6: Real-fixture integration test (VCR-replayed)

**Files:**
- Create: `tests/fixtures/s_filings/sample_s3_excerpt.txt`
- Create: `tests/integration/test_extract_worker_s_filings_vcr.py`
- Create: `tests/integration/cassettes/test_extract_worker_s_filings_vcr/test_real_s3.yaml`

- [ ] **Step 6.1: Add the S-3 fixture text**

Hand-trim a real S-3 cover + use-of-proceeds section to ~5KB and save to
`tests/fixtures/s_filings/sample_s3_excerpt.txt`. The fixture must
contain at least one literal dilution phrase the worker can quote
verbatim (e.g., "common stock offering of $50 million").

(Source: pick any small biotech S-3 from EDGAR; commit only after
confirming the text contains no proprietary content beyond what's in
the public filing.)

- [ ] **Step 6.2: Write the integration test (records a VCR cassette on first run)**

```python
# tests/integration/test_extract_worker_s_filings_vcr.py
"""End-to-end S-1/S-3 worker test, VCR-replayed against a real Anthropic
response (Issue #11 AC: "extracts a real S-3 ... and produces a frozen
SFilingOutput that passes citation-grounding validation").

The cassette is recorded once against the live API (with credentials);
CI replays it. Recording: ANTHROPIC_API_KEY=... uv run pytest -m integration
tests/integration/test_extract_worker_s_filings_vcr.py --record-mode=once
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_research.extract.workers.s_filings import extract_s_filing


FIXTURE = Path(__file__).parent.parent / "fixtures" / "s_filings" / "sample_s3_excerpt.txt"


@pytest.mark.integration
def test_real_s3_passes_citation_grounding(tmp_path: Path, vcr_cassette: object) -> None:
    raw = FIXTURE.read_text()
    out = extract_s_filing(
        raw_doc=raw,
        doc_id="vcr-real-s3",
        cache_root=tmp_path,
        quarantine_root=tmp_path / "quarantine",
    )
    assert out is not None, "real S-3 must pass citation grounding"
    assert out.form_type in ("S-1", "S-3")
    assert out.dilution_event is not None
    # Every citation in the output aligns with the source — validate_or_quarantine
    # returned `out` instead of `None`, which is the contract.
```

- [ ] **Step 6.3: Record the cassette**

```bash
# Requires ANTHROPIC_API_KEY in env. One-time recording; commits the cassette.
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY uv run pytest \
  tests/integration/test_extract_worker_s_filings_vcr.py \
  -m integration --record-mode=once -v
```

Verify the cassette redacts the `x-api-key` header (the `tests/integration/conftest.py`
machinery should already do this — confirm by `grep -i 'api-key' tests/integration/cassettes/*/test_real_s3.yaml` returns nothing sensitive).

- [ ] **Step 6.4: Replay (no key needed)**

```bash
uv run pytest tests/integration/test_extract_worker_s_filings_vcr.py -m integration -v
# Expected: passes from cassette
```

- [ ] **Step 6.5: Commit**

```bash
git add tests/fixtures/s_filings/sample_s3_excerpt.txt \
        tests/integration/test_extract_worker_s_filings_vcr.py \
        tests/integration/cassettes/
git commit -m "test(extract): real-S-3 VCR integration for S-1/S-3 worker (#11)"
```

---

### Task 7: `scripts/promote_prompt.py` + gold set

**Files:**
- Create: `eval/baselines/s_filings_dilution__gold.json`
- Create: `scripts/promote_prompt.py`
- Test: `tests/unit/test_promote_prompt.py`

- [ ] **Step 7.1: Create the gold set**

```json
// eval/baselines/s_filings_dilution__gold.json
{
  "prompt_name": "s_filings_dilution",
  "thresholds": {
    "min_f1": 0.7,
    "max_usd_per_doc": 0.05
  },
  "samples": [
    {
      "doc_id": "gold-001",
      "raw_doc": "This shelf takedown of $200 million of common stock will be used for general corporate purposes and to fund the Phase II clinical trial.",
      "expected": {
        "form_type": "S-3",
        "dilution_event_quote": "shelf takedown of $200 million of common stock",
        "use_of_proceeds_phrases": ["general corporate purposes", "Phase II clinical trial"]
      }
    },
    {
      "doc_id": "gold-002",
      "raw_doc": "We are conducting an at-the-market offering of up to $50 million of common stock. Proceeds will fund commercialization activities.",
      "expected": {
        "form_type": "S-3",
        "dilution_event_quote": "at-the-market offering of up to $50 million of common stock",
        "use_of_proceeds_phrases": ["commercialization activities"]
      }
    }
  ]
}
```

(Gold-set expansion to ~20 samples is followup work; the *contract* and *script* land here.)

- [ ] **Step 7.2: Write failing tests for the promotion script**

```python
# tests/unit/test_promote_prompt.py
"""Unit tests for `scripts/promote_prompt.py` (Issue #11)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.promote_prompt import (
    compute_f1,
    promote,
    PromotionResult,
)


def test_compute_f1_exact_match() -> None:
    expected = {"use_of_proceeds_phrases": ["A", "B"]}
    actual = {"use_of_proceeds_phrases": ["A", "B"]}
    assert compute_f1(expected, actual) == pytest.approx(1.0)


def test_compute_f1_partial_match() -> None:
    expected = {"use_of_proceeds_phrases": ["A", "B"]}
    actual = {"use_of_proceeds_phrases": ["A", "C"]}
    # 1 TP, 1 FP, 1 FN → P=0.5, R=0.5, F1=0.5
    assert compute_f1(expected, actual) == pytest.approx(0.5)


def test_promote_refuses_below_f1_threshold(tmp_path: Path) -> None:
    gold = {
        "prompt_name": "s_filings_dilution",
        "thresholds": {"min_f1": 0.9, "max_usd_per_doc": 1.0},
        "samples": [],  # no samples → F1 = 0.0 → fail
    }
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold))
    client = MagicMock()
    result = promote(
        prompt_name="s_filings_dilution",
        version="v1",
        gold_path=gold_path,
        worker_fn=lambda raw, doc_id: None,  # never called
        langfuse_client=client,
    )
    assert isinstance(result, PromotionResult)
    assert result.promoted is False
    assert "below f1 threshold" in result.reason.lower()
    client.update_prompt.assert_not_called()


def test_promote_flips_tag_when_threshold_met(tmp_path: Path) -> None:
    # Construct a gold set where the worker_fn always returns the expected output:
    sample = {
        "doc_id": "g1",
        "raw_doc": "x",
        "expected": {"form_type": "S-3", "dilution_event_quote": "x", "use_of_proceeds_phrases": []},
    }
    gold = {
        "prompt_name": "s_filings_dilution",
        "thresholds": {"min_f1": 0.5, "max_usd_per_doc": 1.0},
        "samples": [sample],
    }
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold))

    # worker_fn returns a stand-in `SFilingOutput`-like dict matching expected
    def fake_worker(raw: str, doc_id: str) -> dict:
        return {
            "form_type": "S-3",
            "dilution_event": {"citation": {"source_quote": "x"}},
            "use_of_proceeds": [],
        }

    client = MagicMock()
    result = promote(
        prompt_name="s_filings_dilution",
        version="v1",
        gold_path=gold_path,
        worker_fn=fake_worker,
        langfuse_client=client,
    )
    assert result.promoted is True
    client.update_prompt.assert_called_once()
    assert "production" in client.update_prompt.call_args.kwargs["new_labels"]
```

- [ ] **Step 7.3: Run, confirm fail**

```bash
uv run pytest tests/unit/test_promote_prompt.py -v
# Expected: ImportError
```

- [ ] **Step 7.4: Implement the script**

```python
# scripts/promote_prompt.py
"""Eval-gated promotion of a prompt version to the Langfuse `production` tag.

Pattern (Issue #11): a *script*, not a CI pipeline. For a single-machine
research project, a 30-50 line gate is the right level — same discipline
("don't promote unless evals pass"), no Jenkins/Actions theatre.

Usage:

    uv run python scripts/promote_prompt.py s_filings_dilution v1

Reads `eval/baselines/<prompt_name>__gold.json`, runs the matching worker
against each sample under the candidate version, computes a token-level
F1 over the expected fields, and refuses to flip the Langfuse
`production` tag if F1 < threshold or per-doc cost > ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Late-imported workers — keeps the script importable for unit tests that
# pass a synthetic `worker_fn` and don't want to depend on the full extract
# stack at import time.


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    f1: float
    usd_per_doc: float
    reason: str


def compute_f1(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    """Bag-of-phrases F1 across the comparable string fields.

    For v1 we compare two field sets:
    - `dilution_event_quote` (single string; either a TP or an FN/FP)
    - `use_of_proceeds_phrases` (list of strings)

    Future workers can extend the comparator without changing the
    promotion contract — only the F1 implementation grows.
    """
    expected_set: set[str] = set()
    actual_set: set[str] = set()

    if "dilution_event_quote" in expected:
        expected_set.add(("de", expected["dilution_event_quote"]))  # type: ignore[arg-type]
    if "use_of_proceeds_phrases" in expected:
        for p in expected["use_of_proceeds_phrases"]:
            expected_set.add(("up", p))  # type: ignore[arg-type]

    actual_de = actual.get("dilution_event", {}).get("citation", {}).get("source_quote")
    if actual_de:
        actual_set.add(("de", actual_de))
    for c in actual.get("use_of_proceeds", []) or []:
        q = c.get("citation", {}).get("source_quote") if isinstance(c, dict) else None
        if q:
            actual_set.add(("up", q))

    tp = len(expected_set & actual_set)
    fp = len(actual_set - expected_set)
    fn = len(expected_set - actual_set)
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def promote(
    *,
    prompt_name: str,
    version: str,
    gold_path: Path,
    worker_fn: Callable[[str, str], Any],
    langfuse_client: Any,
) -> PromotionResult:
    """Run the gate and (maybe) flip the production tag."""
    gold = json.loads(gold_path.read_text())
    samples = gold.get("samples", [])
    thresholds = gold["thresholds"]
    min_f1 = float(thresholds["min_f1"])
    max_usd = float(thresholds["max_usd_per_doc"])

    if not samples:
        return PromotionResult(
            promoted=False, f1=0.0, usd_per_doc=0.0,
            reason=f"no samples in gold set — below f1 threshold {min_f1}",
        )

    f1_scores: list[float] = []
    for sample in samples:
        actual = worker_fn(sample["raw_doc"], sample["doc_id"])
        actual_dict = actual if isinstance(actual, dict) else actual.model_dump(mode="json")
        f1_scores.append(compute_f1(sample["expected"], actual_dict))
    mean_f1 = sum(f1_scores) / len(f1_scores)

    # v1: no live USD calculation (worker handles caching, the script doesn't
    # want to re-invoke the cost-cap layer). Reserve the ceiling check for
    # when we add per-call cost telemetry from the cache record.
    usd_per_doc = 0.0

    if mean_f1 < min_f1:
        return PromotionResult(
            promoted=False, f1=mean_f1, usd_per_doc=usd_per_doc,
            reason=f"f1={mean_f1:.3f} below f1 threshold {min_f1}",
        )
    if usd_per_doc > max_usd:
        return PromotionResult(
            promoted=False, f1=mean_f1, usd_per_doc=usd_per_doc,
            reason=f"usd_per_doc={usd_per_doc:.4f} above ceiling {max_usd}",
        )

    langfuse_client.update_prompt(
        name=prompt_name,
        version=version,
        new_labels=["production"],
    )
    return PromotionResult(
        promoted=True, f1=mean_f1, usd_per_doc=usd_per_doc,
        reason=f"promoted: f1={mean_f1:.3f} ≥ {min_f1}",
    )


def _resolve_worker(prompt_name: str) -> Callable[[str, str], Any]:
    """Pick the worker function for `prompt_name`. Single dispatch table —
    grows as workers land."""
    if prompt_name == "s_filings_dilution":
        from auto_research.extract.workers.s_filings import extract_s_filing

        def _w(raw: str, doc_id: str) -> Any:
            return extract_s_filing(raw_doc=raw, doc_id=doc_id)

        return _w
    raise ValueError(f"unknown prompt_name: {prompt_name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_name")
    parser.add_argument("version")
    args = parser.parse_args(argv)

    from langfuse import Langfuse  # local import — keeps tests langfuse-free
    client = Langfuse()
    gold_path = Path("eval/baselines") / f"{args.prompt_name}__gold.json"

    result = promote(
        prompt_name=args.prompt_name,
        version=args.version,
        gold_path=gold_path,
        worker_fn=_resolve_worker(args.prompt_name),
        langfuse_client=client,
    )
    log_path = Path("eval/promotions") / f"{args.prompt_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({
            "version": args.version,
            "promoted": result.promoted,
            "f1": result.f1,
            "usd_per_doc": result.usd_per_doc,
            "reason": result.reason,
        }) + "\n")
    print(result.reason)
    return 0 if result.promoted else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["PromotionResult", "compute_f1", "promote"]
```

- [ ] **Step 7.5: Run tests, expect green**

```bash
uv run pytest tests/unit/test_promote_prompt.py -v
```

- [ ] **Step 7.6: Commit**

```bash
git add scripts/promote_prompt.py eval/baselines/ tests/unit/test_promote_prompt.py
git commit -m "feat(extract): eval-gated prompt promotion script (#11)"
```

---

### Task 8: Extend `bump-prompt-version` skill with schema co-versioning check

**Files:**
- Modify: `.claude/skills/bump-prompt-version/SKILL.md`

- [ ] **Step 8.1: Add check #6 to the "Mandatory checks" section**

Insert after the existing check #5 (eval baseline):

```markdown
**6. Schema co-versioning — any output-model field change requires a partnered prompt-version bump:**

If `src/auto_research/extract/schemas.py` is in the diff AND the change touches a Pydantic field shape (field added, removed, type changed), the partnered worker's `*_PROMPT_VERSION` MUST also bump. The cache key includes `SCHEMA_VERSION`, but the Langfuse audit trail anchors on `prompt_version` — both must move together so a single grep of either dimension surfaces the change.

```bash
# Does the diff touch schemas.py?
if git diff --name-only | grep -q "src/auto_research/extract/schemas.py"; then
  # Did any Pydantic field declaration change? (heuristic: lines that look
  # like `name: type` inside a class body)
  if git diff src/auto_research/extract/schemas.py | rg -P '^[+-]\s+\w+:\s+\w' ; then
    # Then SOME `*_PROMPT_VERSION` in extract/prompts/ must also bump.
    if ! git diff src/auto_research/extract/prompts/ | rg -P '^[+-].*_PROMPT_VERSION\s*=' ; then
      echo "FAIL: schemas.py field change without a partnered prompt-version bump."
      echo "      Bump the *_PROMPT_VERSION of the worker(s) whose output model changed."
      exit 1
    fi
  fi
fi
```

If a schema change is genuinely *cosmetic* (rename a comment, reorder methods), the regex skips it. If the change adds/removes/renames a field, the partnered prompt-version bump is mandatory. `SCHEMA_VERSION` on the affected model should also bump in the same commit; that's enforced by code review, not by this grep (regex can't reliably detect a `ClassVar` bump).
```

- [ ] **Step 8.2: Update the pre-submit checklist**

Add a line:

```markdown
- [ ] If `extract/schemas.py` changed a field, the partnered `*_PROMPT_VERSION` and the model's `SCHEMA_VERSION` both bumped in the same commit.
```

- [ ] **Step 8.3: Commit**

```bash
git add .claude/skills/bump-prompt-version/SKILL.md
git commit -m "chore(skill): bump-prompt-version enforces schema co-versioning (#11)"
```

---

### Task 9: AGENTS.md INV-6 + CONTRACTS.md wording updates

**Files:**
- Modify: `AGENTS.md` (around line 82)
- Modify: `docs/CONTRACTS.md` (around lines 90-92)

- [ ] **Step 9.1: Update INV-6 in AGENTS.md**

Replace the existing INV-6 block (around lines 82-86) with:

```markdown
**INV-6. Determinism: completion configs are version-pinned.** Extraction
workers are
`(raw_doc, prompt_version, schema_version, model_id, decoding_params) → ExtractionOutput`
pure functions with content-hash idempotent cache. Prompt registry lives in
Langfuse; the prompt and output-schema versions are colocated in code
(`extract/prompts/<name>.py` and `extract/schemas.py` `ClassVar`). Changing
any of the five inputs without invalidating the cache key silently corrupts
outputs. The `bump-prompt-version` skill defends prompt + schema co-versioning;
the cache key itself (`extract/cache.py`) defends `model_id` and `decoding_params`.
Promotion to the Langfuse `production` tag is gated by
`scripts/promote_prompt.py` — eval-gated, not a manual flip.
```

- [ ] **Step 9.2: Update `docs/CONTRACTS.md` lines 90-92**

Find the block:

```markdown
- Worker functions are `(raw_doc: RawDoc, prompt_version: str) → Output` —
  pure functions. Content-hash cached on `sha256(raw_doc.bytes + prompt_version)`.
```

Replace with:

```markdown
- Worker functions are
  `(raw_doc: RawDoc, prompt_version: str, schema_version: str, model_id: str, decoding_params: dict) → Output`
  — pure functions. Content-hash cached on
  `sha256(raw_doc.bytes + prompt_version + schema_version + model_id + canonical_json(decoding_params))`
  (see `src/auto_research/extract/cache.py`).
```

- [ ] **Step 9.3: Commit**

```bash
git add AGENTS.md docs/CONTRACTS.md
git commit -m "docs(invariants): tighten INV-6 to completion-config-pinned (#11)"
```

---

### Task 10: `.gitignore` + README section

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 10.1: Ensure cache + quarantine dirs are gitignored**

```bash
grep -q "^data/cache/" .gitignore || echo "data/cache/" >> .gitignore
grep -q "^data/quarantine/" .gitignore || echo "data/quarantine/" >> .gitignore
grep -q "^eval/promotions/" .gitignore || echo "eval/promotions/" >> .gitignore
```

- [ ] **Step 10.2: Add a README section**

Append (or insert at the appropriate spot in `README.md`):

```markdown
### Prompt lifecycle

Prompts live as `<name>_PROMPT` constants in `src/auto_research/extract/prompts/`,
colocated with a `<NAME>_PROMPT_VERSION = "vN"` tag. Code is the source of
truth at runtime; Langfuse holds the registry for version history and tag
state. The discipline:

1. **Edit** a prompt file → the `bump-prompt-version` skill blocks the
   commit unless `*_PROMPT_VERSION` also bumps. If the partnered Pydantic
   output model's fields changed, the skill also requires bumping the
   model's `SCHEMA_VERSION`.
2. **Cache** is keyed on the full completion config —
   `(raw_doc, prompt_version, schema_version, model_id, decoding_params)`
   — so model swaps and decoding changes don't silently reuse stale entries.
3. **Promote** to the Langfuse `production` tag via
   `uv run python scripts/promote_prompt.py <prompt_name> <version>` — the
   script runs the gold-set eval and refuses the tag flip below the F1
   threshold or above the cost ceiling.

See `AGENTS.md` INV-6 for the formal invariant.
```

- [ ] **Step 10.3: Commit**

```bash
git add .gitignore README.md
git commit -m "docs(readme): prompt-lifecycle section + gitignore caches (#11)"
```

---

### Task 11: Full verification

- [ ] **Step 11.1: `make quick`**

```bash
make quick
# Expected: ruff + mypy clean. Fix any lint/type issues before proceeding.
```

- [ ] **Step 11.2: Full unit + integration suite**

```bash
uv run pytest tests/unit/ -v
uv run pytest tests/integration/test_extract_worker_s_filings_vcr.py -m integration -v
# Expected: all green
```

- [ ] **Step 11.3: Skill self-check**

Run the `bump-prompt-version` skill's grep checks against the diff:

```bash
git diff origin/main -- src/auto_research/extract/prompts/ | rg -P '^[+-].*_PROMPT_VERSION\s*='
# Expected: shows the +S_FILINGS_DILUTION_PROMPT_VERSION = "v1" line (new file, no - line is fine)
```

- [ ] **Step 11.4: Open the PR**

```bash
git push -u origin feat/11-prompts-registry-s1s3
gh pr create --title "feat(extract): prompts registry + S-1/S-3 worker + lifecycle discipline (#11)" \
  --body "$(cat <<'EOF'
Closes #11.

## Summary
- `extract/prompts/` convention landed: one file per prompt, version
  constant colocated, Langfuse registry as the version-history sidecar.
- S-1/S-3 worker end-to-end: prompt → Anthropic → JSON → SFilingOutput
  → citation grounding → cache. First validator of the full pipeline.
- Lifecycle additions per the 2026-05-24 discussion:
  1. Cache key tightened to `(raw_doc, prompt_version, schema_version, model_id, decoding_params)`. INV-6 wording updated.
  2. `SCHEMA_VERSION: ClassVar` on every output model; `bump-prompt-version` skill enforces co-versioning.
  3. `scripts/promote_prompt.py` gates Langfuse `production` tag flips on a gold-set F1 threshold.

## Change Contract
- Tier: 2
- Problem: First end-to-end extraction worker + the prompt-lifecycle discipline the rest of the extraction stack depends on.
- Scope: `extract/prompts/`, `extract/cache.py`, `extract/workers/s_filings.py`, `extract/schemas.py` (add ClassVar), `scripts/promote_prompt.py`, skill extension, INV-6 wording.
- Invariants touched: INV-2 (citation grounding — verified by `test_corrupted_citation_routes_to_quarantine`), INV-6 (tightened from prompt-pinned to completion-config-pinned).
- Verification: unit suite green, VCR integration green, `bump-prompt-version` skill checks pass.
- Rollback: `git revert` — additive feature, no migrations.

## AC mapping
- S-1/S-3 worker extracts real S-3, passes citation grounding → `tests/integration/test_extract_worker_s_filings_vcr.py::test_real_s3_passes_citation_grounding`
- Corrupted citation routes to quarantine → `tests/unit/test_extract_worker_s_filings.py::test_corrupted_citation_routes_to_quarantine`
- Prompt version constant colocated + Langfuse-registered → `tests/unit/test_extract_prompts.py::test_prompt_version_is_human_readable_tag`, `tests/unit/test_extract_prompts_registry.py::test_register_prompt_calls_langfuse_create_prompt`
- Cache key captures full completion config (5 inputs) → `tests/unit/test_extract_cache.py::test_cache_key_changes_when_any_completion_config_input_changes`
- SCHEMA_VERSION co-versioning enforced → `tests/unit/test_extract_schemas.py::test_every_output_model_carries_schema_version` + skill check #6
- Cache hit returns identical output without LLM call → `tests/unit/test_extract_worker_s_filings.py::test_cache_hit_skips_llm_call`
- promote_prompt.py gates production tag → `tests/unit/test_promote_prompt.py::test_promote_refuses_below_f1_threshold` and `::test_promote_flips_tag_when_threshold_met`
- `bump-prompt-version` skill checks pass → run manually in Task 11.3

## Sensitive-path evidence
| Row | Form |
|---|---|
| Unit test name + green | `tests/unit/test_extract_worker_s_filings.py::*` (3 passed) |
| DeepEval score delta | N/A — DeepEval not adopted in this issue; bag-of-phrases F1 used in `promote_prompt.py`. DeepEval scaffold is followup work. |

## Doc-sync
- Updated `AGENTS.md` INV-6 and `docs/CONTRACTS.md` §1 wording to match the tightened cache key.
- New README section "Prompt lifecycle" pointing at INV-6 and the workflow.
EOF
)"
```

---

## Self-review

**Spec coverage** — every AC mapped:
- [x] S-3 extraction + citation grounding → Task 5 + Task 6
- [x] Corrupted citation → quarantine → Task 5 (test 3)
- [x] Prompt version constant colocated → Task 1
- [x] Cache key captures full completion config + multi-model test → Task 3
- [x] SCHEMA_VERSION co-versioning → Task 2 + Task 8
- [x] Cache hit skips LLM → Task 5 (test 2)
- [x] `promote_prompt.py` gates production tag → Task 7
- [x] `bump-prompt-version` skill checks pass → Task 8 + Task 11

**Placeholder scan** — no TBDs; every step has runnable code or commands. The S-3 fixture text and VCR cassette are the only non-code artifacts; both have explicit recording instructions.

**Type consistency** — `cache_key(*, raw_doc, prompt_version, schema_version, model_id, decoding_params)` signature is identical between Task 3 (implementation) and Task 5 (caller); `SFilingOutput.SCHEMA_VERSION` matches between Task 2 (definition) and Task 5 (consumer); `extract_s_filing(*, raw_doc, doc_id, cache_root=None, quarantine_root=None, anthropic_client=None)` matches between worker and tests.

**Commit boundary check** — 10 commits, each touches one logical concern. Reviewable independently.
