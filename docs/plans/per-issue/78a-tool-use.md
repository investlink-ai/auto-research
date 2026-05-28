# PR-A (issue #78): tool_use migration + categorical Claim.confidence

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Anthropic API calls to server-side schema-validated `tool_use`, and migrate `Claim.confidence` from `float` to `Literal['high','medium','low']`.

**Architecture:**
- `ExtractionFn` protocol gains `output_schema: type[BaseModel]` kwarg.
- `client._call` sends `tools=[{name:'record_extraction', input_schema:<pydantic schema>}]` + `tool_choice={'type':'tool','name':'record_extraction'}` in lieu of free-form text. Server-side schema-shape enforcement cuts JSON-decode + schema-noncompliance quarantines from ~5-15% to <1%.
- `_common.run_single_shot_extraction` extracts `tool_use.input` (dict) directly. Deletes `_FENCE_RE`, `_strip_fence`, `json.loads`, and the json-decode quarantine branch.
- `_resolve_spans` and `validate_or_quarantine` paths unchanged — they remain the citation-grounding boundary (INV-2).
- `Claim.confidence: float` → `Literal['high','medium','low']` across schema + 4 prompts + 11 test fixtures.
- **No `PROMPT_VERSION` / `SCHEMA_VERSION` bump** — pre-deployment policy; no downstream consumer has adopted the artifact yet.
- **Anthropic native citations are deliberately not adopted** — issue trade-off; `_resolve_spans` + AMBIGUOUS handling stays as the citation mechanism.

**Tech Stack:** Anthropic Python SDK (>=0.104), pydantic v2 (`model_json_schema`), OpenTelemetry, pytest.

**Tier classification:** Tier 2 — `extract/schemas.py` is a sensitive path (AGENTS.md §3). Requires failing test first, full pytest module suite green, PR body citing test names.

---

## Task 1: Add `ToolUseBlock` test helper + switch one client test fixture

**Files:**
- Modify: `tests/unit/conftest.py` (add helper)
- Test: existing tests will be rewritten in Task 4; this task only proves the helper

The current tests build a `Message` with a `TextBlock`. After migration, every Anthropic response in our test suite must contain a `ToolUseBlock` carrying the parsed dict as `.input`. We add a single helper to keep the construction one-line at every call site.

- [ ] **Step 1: Read current conftest fixture shape**

```bash
sed -n '1,50p' tests/unit/conftest.py
```

- [ ] **Step 2: Add `make_tool_use_message` helper to `tests/unit/conftest.py`**

```python
# Append to tests/unit/conftest.py
from anthropic.types import Message, ToolUseBlock, Usage


def make_tool_use_message(
    *,
    tool_input: dict,
    tool_name: str = "record_extraction",
    model: str = "claude-haiku-4-5-20251001",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> Message:
    """Build a Message whose content is a single ToolUseBlock.

    Mirrors the on-wire shape after PR-A: every extraction response
    carries a tool_use block whose `.input` is the parsed dict —
    skipping the text-+-json.loads round-trip the old path required.
    """
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=[
            ToolUseBlock(
                type="tool_use",
                id="toolu_test",
                name=tool_name,
                input=tool_input,
            )
        ],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )
```

- [ ] **Step 3: Run pytest collection to verify import-clean**

```bash
uv run pytest tests/unit/conftest.py --collect-only 2>&1 | tail -10
```

Expected: no collection errors.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/conftest.py
git commit -m "test(extract): add make_tool_use_message helper for tool_use migration"
```

---

## Task 2: Extend `ExtractionFn` protocol with `output_schema`

**Files:**
- Modify: `src/auto_research/extract/client.py` (Protocol signature, `_call` signature)
- Test: `tests/unit/test_extract_client.py` (failing test first)

The Protocol gains a required kwarg `output_schema: type[BaseModel]`. Every call site passes it. This is the contract change that drives the rest of the PR.

- [ ] **Step 1: Read current client.py**

```bash
sed -n '1,200p' src/auto_research/extract/client.py
```

- [ ] **Step 2: Write failing test in `tests/unit/test_extract_client.py`**

Add this test (do not touch existing tests yet; they get rewritten in Task 4):

```python
def test_make_extraction_client_sends_tool_use_payload(monkeypatch):
    """The client must send the output schema as a tools[input_schema]
    and force tool_choice; the response's tool_use.input must reach
    callers as the parsed dict.
    """
    from pydantic import BaseModel
    from auto_research.extract.client import make_extraction_client

    class TinyOutput(BaseModel):
        SCHEMA_VERSION = "v1"
        answer: str

    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            from tests.unit.conftest import make_tool_use_message
            return make_tool_use_message(tool_input={"answer": "42"})

    class FakeAnthropic:
        def __init__(self):
            self.messages = FakeMessages()

    client = make_extraction_client(
        worker="s_filings", anthropic_client=FakeAnthropic()
    )
    resp = client(
        task="dilution_event",
        system_prompt="sys",
        user_content="user",
        output_schema=TinyOutput,
    )

    # tool_use payload
    assert "tools" in captured
    assert captured["tools"][0]["name"] == "record_extraction"
    assert captured["tools"][0]["input_schema"]["type"] == "object"
    assert "answer" in captured["tools"][0]["input_schema"]["properties"]
    assert captured["tool_choice"] == {
        "type": "tool", "name": "record_extraction"
    }

    # Response is the Message with tool_use block
    assert resp.content[0].type == "tool_use"
    assert resp.content[0].input == {"answer": "42"}
```

- [ ] **Step 3: Run test, confirm it fails**

```bash
uv run pytest tests/unit/test_extract_client.py::test_make_extraction_client_sends_tool_use_payload -xvs
```

Expected: TypeError (unexpected kwarg `output_schema`) or KeyError on `tools`.

- [ ] **Step 4: Update `ExtractionFn` Protocol and `_call` signature in `src/auto_research/extract/client.py`**

```python
# src/auto_research/extract/client.py
# In imports, add:
from pydantic import BaseModel

# Replace ExtractionFn Protocol:
class ExtractionFn(Protocol):
    """Type of the callable returned by `make_extraction_client`.

    Documented as a Protocol so workers can annotate `_CLIENT: ExtractionFn`
    without having to spell out the kwargs each time. `output_schema` is
    sent to Anthropic as the tool's `input_schema`; the response carries
    a single `tool_use` block whose `.input` is the parsed dict.
    """

    def __call__(
        self,
        *,
        task: str,
        system_prompt: str,
        user_content: str,
        output_schema: type[BaseModel],
        max_tokens: int = ...,
    ) -> Message: ...
```

Then update the `_call` definition inside `make_extraction_client`:

```python
    def _call(
        *,
        task: str,
        system_prompt: str,
        user_content: str,
        output_schema: type[BaseModel],
        max_tokens: int = 4096,
    ) -> Message:
        model = route_model(worker, task)

        extra_kwargs: dict[str, Any] = {}
        if model.startswith(("claude-sonnet-", "claude-opus-")):
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": EXTENDED_THINKING_BUDGET,
            }

        tool = {
            "name": "record_extraction",
            "description": (
                "Emit the structured extraction result. The model must "
                "call this tool exactly once; its input is validated "
                "against the worker's pydantic schema downstream."
            ),
            "input_schema": output_schema.model_json_schema(),
        }

        response = sdk.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=cached_system_block(system_prompt),  # type: ignore[arg-type]
            messages=[{"role": "user", "content": user_content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "record_extraction"},
            **extra_kwargs,
        )

        # OTel attributes unchanged from before
        span = trace.get_current_span()
        span.set_attribute("llm.cost.est_usd", usd_for_message(response))
        usage = response.usage
        span.set_attribute("llm.input_tokens", usage.input_tokens)
        span.set_attribute("llm.output_tokens", usage.output_tokens)
        if usage.cache_creation_input_tokens is not None:
            span.set_attribute(
                "llm.cache_creation_input_tokens",
                usage.cache_creation_input_tokens,
            )
        if usage.cache_read_input_tokens is not None:
            span.set_attribute(
                "llm.cache_read_input_tokens", usage.cache_read_input_tokens
            )

        return response
```

- [ ] **Step 5: Run new test; verify pass**

```bash
uv run pytest tests/unit/test_extract_client.py::test_make_extraction_client_sends_tool_use_payload -xvs
```

Expected: PASS.

- [ ] **Step 6: Do NOT commit yet** — existing client tests now fail; Task 4 will fix them in one commit.

---

## Task 3: Switch `run_single_shot_extraction` to parse `tool_use.input`

**Files:**
- Modify: `src/auto_research/extract/workers/_common.py`
- Test: `tests/unit/test_extract_worker_common.py` (failing test first, then rewrite)

The function loses the `_strip_fence` → `json.loads` path and instead pulls `tool_use.input` (already a dict). The json-decode quarantine branch is deleted entirely. Schema-validation quarantine, `_resolve_spans` quarantine, and `validate_or_quarantine` paths are unchanged.

- [ ] **Step 1: Write failing test**

Add to `tests/unit/test_extract_worker_common.py`:

```python
def test_run_single_shot_extraction_extracts_tool_use_input(tmp_path, monkeypatch):
    """End-to-end: tool_use.input is the parsed dict; no json.loads."""
    from pydantic import BaseModel
    from auto_research.extract.workers import _common

    class TinyOutput(BaseModel):
        SCHEMA_VERSION = "v1"
        answer: str
        def model_dump(self, *args, **kwargs):
            return {"answer": self.answer}

    from tests.unit.conftest import make_tool_use_message

    def fake_client(**kwargs):
        assert "output_schema" in kwargs
        return make_tool_use_message(tool_input={"answer": "42"})

    def fake_factory(_):
        return fake_client

    result = _common.run_single_shot_extraction(
        raw_doc="raw text 42",
        doc_id="d1",
        worker="s_filings",
        task="dilution_event",
        prompt="sys prompt",
        prompt_version="v1",
        output_model=TinyOutput,
        max_tokens=1024,
        cache_root=tmp_path / "cache",
        quarantine_root=tmp_path / "q",
        client_factory=fake_factory,
    )
    assert result is not None
    assert result.answer == "42"
```

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/test_extract_worker_common.py::test_run_single_shot_extraction_extracts_tool_use_input -xvs
```

Expected: AssertionError on `"output_schema" in kwargs` (the function doesn't pass it yet) or AttributeError on `b.text` (the tool_use block has no `.text`).

- [ ] **Step 3: Rewrite the parsing block in `_common.run_single_shot_extraction`**

In `src/auto_research/extract/workers/_common.py`:

a) Delete `_FENCE_RE` and `_strip_fence` entirely (lines defining both, plus the import of `re` if it becomes unused — check first).

b) Update the call to `client(...)`:

```python
        try:
            response = client(
                task=task,
                system_prompt=prompt,
                user_content=raw_doc,
                output_schema=output_model,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            span.set_attribute("extract.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, _truncate(str(exc))))
            raise
```

c) Replace the text-extraction-+-json.loads block with tool_use extraction. Delete the entire `text = _strip_fence(...)` and `try: parsed = json.loads(text) except json.JSONDecodeError ...` block. Replace with:

```python
        tool_use_blocks = [
            b for b in response.content
            if b.type == "tool_use" and b.name == "record_extraction"
        ]
        if not tool_use_blocks:
            _write_quarantine(
                quarantine_root=quarantine_root,
                worker=worker,
                prompt_version=prompt_version,
                doc_id=doc_id,
                parsed={"raw_response_blocks": [
                    {"type": b.type} for b in response.content
                ]},
                error="model emitted no record_extraction tool_use block",
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, "no tool_use block")
            )
            return None
        parsed = tool_use_blocks[0].input
        if not isinstance(parsed, dict):
            _write_quarantine(
                quarantine_root=quarantine_root,
                worker=worker,
                prompt_version=prompt_version,
                doc_id=doc_id,
                parsed={"raw": parsed},
                error=f"tool_use.input is not a dict: {type(parsed).__name__}",
            )
            span.set_attribute("extract.outcome", "quarantined")
            span.set_status(
                Status(StatusCode.ERROR, "tool_use.input not dict")
            )
            return None
```

d) Remove `import json` and `import re` from the top of the file if no longer used (the rest of the file uses neither after this change — `_resolve_spans` uses `re` for `_quote_to_flex_regex`. So keep `re`. Remove `json`).

e) Remove `_strip_fence` and `_FENCE_RE` from `__all__`.

- [ ] **Step 4: Run the new test; verify pass**

```bash
uv run pytest tests/unit/test_extract_worker_common.py::test_run_single_shot_extraction_extracts_tool_use_input -xvs
```

Expected: PASS.

- [ ] **Step 5: Run the full `_common` test module; expect many existing failures**

```bash
uv run pytest tests/unit/test_extract_worker_common.py -x 2>&1 | tail -40
```

Expected: existing tests that used `TextBlock` responses now fail. Task 4 fixes them.

---

## Task 4: Rewrite existing test fixtures from `TextBlock` to `ToolUseBlock`

**Files:**
- Modify: `tests/unit/test_extract_client.py`
- Modify: `tests/unit/test_extract_worker_common.py`
- Modify: `tests/unit/test_extract_worker_s_filings.py`
- Modify: `tests/unit/test_extract_worker_eight_k.py`
- Modify: `tests/unit/test_extract_worker_transcript.py`
- Modify: `tests/unit/test_extract_worker_ten_k.py`

Pure mechanical rewrite: every `TextBlock(type='text', text=json.dumps(payload))` becomes `make_tool_use_message(tool_input=payload)` via the conftest helper. The wrapping `Message(...)` construction goes away.

- [ ] **Step 1: Inventory every fake-response site**

```bash
grep -rn "TextBlock\b" tests/unit/ src/auto_research/
```

- [ ] **Step 2: Per file, replace each occurrence**

For each match:
- Replace `Message(...)` constructor calls that wrap a single `TextBlock` with `make_tool_use_message(tool_input=<the dict>)`. Import `make_tool_use_message` from `tests.unit.conftest` if not already imported.
- Adjust any `fake_client` / `FakeMessages.create` to:
  - Accept `output_schema=` kwarg (assert it's present)
  - Return `make_tool_use_message(tool_input=<dict>)`

Pattern (before):
```python
response = Message(
    id="msg_1",
    type="message",
    role="assistant",
    model="claude-haiku-4-5-20251001",
    content=[TextBlock(type="text", text=json.dumps(payload))],
    stop_reason="end_turn",
    stop_sequence=None,
    usage=Usage(input_tokens=100, output_tokens=50, ...),
)
```

Pattern (after):
```python
from tests.unit.conftest import make_tool_use_message
response = make_tool_use_message(tool_input=payload)
```

- [ ] **Step 3: For tests that exercise the json-decode quarantine path (e.g., a test that returns malformed JSON), delete them or convert to the new "no tool_use block" quarantine path**

```bash
grep -n "json decode" tests/unit/test_extract_worker_common.py tests/unit/test_extract_worker_*.py
```

For each: the json-decode failure mode no longer exists. Either delete (if the test only checked json-decode) or rewrite to test "no tool_use block" / "tool_use.input not a dict".

- [ ] **Step 4: Run the full extract test suite**

```bash
uv run pytest tests/unit/test_extract_client.py tests/unit/test_extract_worker_*.py -x 2>&1 | tail -20
```

Expected: all tests pass (except possibly tests that exercise `Claim.confidence: float`, which Task 6 addresses).

- [ ] **Step 5: Commit the tool_use migration as one logical change**

```bash
git add src/auto_research/extract/client.py src/auto_research/extract/workers/_common.py tests/unit/
git commit -m "$(cat <<'EOF'
feat(extract): switch extraction client to Anthropic tool_use

Server-side schema-shape enforcement via tools[input_schema] + forced
tool_choice replaces the text-JSON path. tool_use.input arrives as a
dict; _strip_fence, _FENCE_RE, and the json-decode quarantine branch
are removed.

Trade-off accepted: loses the option to use Anthropic native citations
(tool_use.input doesn't carry citation metadata). _resolve_spans +
AMBIGUOUS handling stays as the citation mechanism.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update worker call sites to pass `output_schema=`

**Files:**
- Modify: `src/auto_research/extract/workers/s_filings.py`
- Modify: `src/auto_research/extract/workers/eight_k.py`
- Modify: `src/auto_research/extract/workers/transcript.py`
- Modify: `src/auto_research/extract/workers/ten_k.py`

Each worker already has the output model in scope at every `run_single_shot_extraction` call site (as the `output_model=` arg). We don't need to add it because `_common.run_single_shot_extraction` now forwards `output_model` as `output_schema` to the client internally (Task 3 wired this). **Verify no worker calls `client(...)` directly — they all go through `_common`.**

- [ ] **Step 1: Confirm workers don't call client directly**

```bash
grep -n "_CLIENT(" src/auto_research/extract/workers/*.py
grep -n "_get_client(.*)(" src/auto_research/extract/workers/*.py
```

Expected: no matches. All worker `_CLIENT` usage goes through `client_factory=_get_client` into `_common.run_single_shot_extraction`.

- [ ] **Step 2: If a direct call exists**, add `output_schema=<model>` to it.

- [ ] **Step 3: Run worker test modules**

```bash
uv run pytest tests/unit/test_extract_worker_s_filings.py tests/unit/test_extract_worker_eight_k.py tests/unit/test_extract_worker_transcript.py tests/unit/test_extract_worker_ten_k.py -x 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 4: Commit only if changes made**; otherwise skip to Task 6.

---

## Task 6: Migrate `Claim.confidence` to `Literal['high','medium','low']`

**Files:**
- Modify: `src/auto_research/extract/schemas.py`
- Test: `tests/unit/test_extract_schemas.py` (failing test first)

Sensitive path (AGENTS.md §3). Failing test first, full module suite green after, PR cites test name.

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_extract_schemas.py`:

```python
def test_claim_confidence_accepts_literal_values():
    from auto_research.extract.schemas import Claim, Citation
    citation = Citation(source_span=[0, 5], source_quote="hello")
    for level in ("high", "medium", "low"):
        c = Claim(citation=citation, confidence=level, ...)  # fill required fields per current Claim shape
        assert c.confidence == level


def test_claim_confidence_rejects_float():
    from auto_research.extract.schemas import Claim, Citation
    from pydantic import ValidationError
    citation = Citation(source_span=[0, 5], source_quote="hello")
    with pytest.raises(ValidationError):
        Claim(citation=citation, confidence=0.7, ...)  # fill required fields


def test_claim_confidence_rejects_other_strings():
    from auto_research.extract.schemas import Claim, Citation
    from pydantic import ValidationError
    citation = Citation(source_span=[0, 5], source_quote="hello")
    with pytest.raises(ValidationError):
        Claim(citation=citation, confidence="maybe", ...)
```

(Before writing, run `grep -n "class Claim" src/auto_research/extract/schemas.py` and `sed -n '<line>,<line+30>p'` to fill in the required fields list correctly.)

- [ ] **Step 2: Confirm failure**

```bash
uv run pytest tests/unit/test_extract_schemas.py::test_claim_confidence_accepts_literal_values tests/unit/test_extract_schemas.py::test_claim_confidence_rejects_float tests/unit/test_extract_schemas.py::test_claim_confidence_rejects_other_strings -xvs
```

Expected: first test fails (Claim rejects 'high' string because the field is `float`); second/third fail to raise.

- [ ] **Step 3: Update `Claim.confidence` in `src/auto_research/extract/schemas.py`**

Locate the `Claim` model. Change:

```python
    confidence: float = Field(..., ge=0.0, le=1.0)
```

to:

```python
    confidence: Literal["high", "medium", "low"]
```

Add `Literal` to imports from `typing` if not already.

**Do NOT bump SCHEMA_VERSION on TenKOutput/TranscriptOutput/EightKOutput/SFilingOutput** — pre-deployment policy. (Memory: `feedback-prompt-version-bump-policy`.)

- [ ] **Step 4: Run new tests**

```bash
uv run pytest tests/unit/test_extract_schemas.py -xvs 2>&1 | tail -20
```

Expected: new tests pass; existing tests that construct `Claim(confidence=0.x)` fail.

---

## Task 7: Update 11 test-fixture Claim constructions

**Files:**
- Modify: `tests/unit/test_extract_worker_common.py` (1 site)
- Modify: `tests/unit/test_extract_worker_eight_k.py` (2 sites)
- Modify: `tests/unit/test_extract_worker_s_filings.py` (1 site)
- Modify: `tests/unit/test_extract_worker_ten_k.py` (3 sites — narrative + RAG; the 3 financials sites already use `"high"`)
- Modify: `tests/unit/test_extract_worker_transcript.py` (2 sites)

Mapping convention for the literal swap (preserves the rough semantics of the old float):
- `>= 0.7` → `"high"`
- `>= 0.4` → `"medium"`
- `< 0.4` → `"low"`

- [ ] **Step 1: Inventory all float-confidence Claim sites**

```bash
grep -rn "confidence=[0-9]" tests/unit/
```

- [ ] **Step 2: Per-site replacement** — for each match, apply the threshold map above.

- [ ] **Step 3: Run all unit tests**

```bash
uv run pytest tests/unit/ -x 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/auto_research/extract/schemas.py tests/unit/
git commit -m "$(cat <<'EOF'
feat(extract): Claim.confidence -> Literal['high','medium','low']

Removes the last float confidence field on LLM-emitted Claims. Matches
the FinancialLineItem.confidence shape already in use. Float
confidence was uncalibrated noise (memory: llm-confidence-is-categorical).

SupplierMention.resolver_confidence and CustomerMention.resolver_confidence
stay float — they're resolver scores, not LLM judgments.

No SCHEMA_VERSION bump per pre-deployment policy: no downstream consumer
has adopted the artifact yet.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update 4 extraction prompts to instruct categorical confidence

**Files:**
- Modify: `src/auto_research/extract/prompts/s_filings_dilution.py`
- Modify: `src/auto_research/extract/prompts/eight_k.py`
- Modify: `src/auto_research/extract/prompts/ten_k_narrative.py`
- Modify: `src/auto_research/extract/prompts/transcript.py`
- (No change: `ten_k_financials.py` — already categorical.)

In each file, three categories of change:

1. Constraint sentence: `"with confidence in [0, 1]"` → `"with confidence categorical (one of 'high', 'medium', or 'low')"`
2. Schema description: `"(a float in [0, 1])"` → `"(EXACTLY one of \"high\", \"medium\", or \"low\")"`
3. Example JSON: `"confidence": 0.9` → `"confidence": "high"` (and similarly for other example floats; map 0.x → high/medium/low using the same threshold as Task 7).

**Do NOT bump `*_VERSION` strings in these files.** Pre-deployment policy.

- [ ] **Step 1: Per file, apply the three rewrites**

Use sed or manual Edit for each. Pattern for s_filings_dilution.py:

```python
# Read first
cat src/auto_research/extract/prompts/s_filings_dilution.py | head -80
```

Then Edit:
- Line ~46: `"with confidence in [0, 1]"` → `"with confidence categorical (one of \"high\", \"medium\", or \"low\")"`
- Line ~53: `"(a float in [0, 1])"` → `"(EXACTLY one of \"high\", \"medium\", or \"low\")"`
- Line ~61: `"confidence": 0.9` → `"confidence": "high"`

Repeat for the other three files.

- [ ] **Step 2: Verify each prompt's tests still pass**

```bash
uv run pytest tests/unit/test_extract_prompts.py -x
```

- [ ] **Step 3: Confirm no `*_VERSION` was bumped**

```bash
git diff src/auto_research/extract/prompts/ | grep -E "_VERSION"
```

Expected: empty.

- [ ] **Step 4: Commit**

```bash
git add src/auto_research/extract/prompts/
git commit -m "$(cat <<'EOF'
feat(extract): align 4 prompts with categorical Claim.confidence

s_filings_dilution, eight_k, ten_k_narrative, transcript now instruct
the model to emit confidence as one of 'high'/'medium'/'low' instead
of a float in [0,1]. Matches the schema change in the previous commit.

No PROMPT_VERSION bump per pre-deployment policy — no downstream consumer
has adopted this prompt contract yet.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Full suite + open PR

- [ ] **Step 1: `make check`**

```bash
make check 2>&1 | tail -20
```

Expected: lint + mypy + unit tests all green.

- [ ] **Step 2: Push branch**

```bash
git push -u origin feat/78-tool-use
```

- [ ] **Step 3: Open PR**

```bash
gh pr create --base main --title "feat(extract): tool_use migration + categorical Claim.confidence (#78 PR-A)" --body "$(cat <<'EOF'
## Summary

PR-A of #78: switches the extraction client to Anthropic `tool_use` and
migrates `Claim.confidence` from `float` to `Literal['high','medium','low']`.

- `ExtractionFn.__call__` gains `output_schema: type[BaseModel]`; the
  client sends it as the tool's `input_schema` and forces
  `tool_choice={'type':'tool','name':'record_extraction'}`.
- `_common.run_single_shot_extraction` reads `tool_use.input` (dict)
  directly; `_FENCE_RE`, `_strip_fence`, and the json-decode quarantine
  branch are removed.
- 4 prompts (`s_filings_dilution`, `eight_k`, `ten_k_narrative`,
  `transcript`) re-instructed to emit categorical confidence. The
  financials prompt was already categorical.
- 11 test fixtures' float `confidence=` values rewritten using
  threshold map (>=0.7 high; >=0.4 medium; else low).

**Trade-off accepted:** loses the option to use Anthropic native
citations (`tool_use.input` doesn't carry citation metadata).
`_resolve_spans` + AMBIGUOUS handling stays as the citation mechanism.

**No `PROMPT_VERSION` / `SCHEMA_VERSION` bumps** — pre-deployment
policy, no downstream consumer has adopted the artifact yet.

## Tier

Tier 2 — `extract/schemas.py` is a sensitive path (AGENTS.md §3).

## Evidence

- `tests/unit/test_extract_client.py::test_make_extraction_client_sends_tool_use_payload`
- `tests/unit/test_extract_worker_common.py::test_run_single_shot_extraction_extracts_tool_use_input`
- `tests/unit/test_extract_schemas.py::test_claim_confidence_accepts_literal_values`
- `tests/unit/test_extract_schemas.py::test_claim_confidence_rejects_float`
- `tests/unit/test_extract_schemas.py::test_claim_confidence_rejects_other_strings`
- `make check` green.

## Test plan

- [x] `make check` green
- [x] All `tests/unit/test_extract_*.py` pass with tool_use mocks
- [x] No `*_VERSION` bumps anywhere in the diff

Closes part A of #78. PR-B and PR-C tracked separately.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

- **Spec coverage:** every bullet in issue §A (tool_use migration + categorical confidence) maps to a task: Tasks 1-5 are tool_use; Tasks 6-8 are categorical; Task 9 is verification + PR.
- **Cache invariance:** no `*_VERSION` bumps; per memory `feedback-prompt-version-bump-policy`. Pre-existing local cache files will fail validation against new categorical schema → quarantine, which is the correct behavior in pre-deployment.
- **Sensitive-path discipline:** `extract/schemas.py` change covered by failing tests in `test_extract_schemas.py` before implementation.
- **No tradeable-flag drift, no implicit-backend, no unit-test-makes-real-API-call** — none of these patterns apply.
- **`bump-prompt-version` skill applicability:** the skill would normally fire on prompt edits. The skill is mechanical — it bumps versions. The pre-deployment policy in memory says don't bump now. Memory rule wins (user instructions > skills); commit message documents this.
