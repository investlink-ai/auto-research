# Testing for AI engineering — a tutorial

> Self-study material. Companion to
> [`observability-for-ai-engineering.md`](observability-for-ai-engineering.md).
> Target reader: an engineer who can write a pytest fixture and wants the
> *senior-judgment* layer on top. Interview-prep depth.

---

## 1. Why testing AI/ML systems is uniquely hard

A web-app test asks "given input X, did the function return Y?" A few axes where
LLM/ML systems break that simple model:

| Concern | Why it's hard | Mitigation |
|---|---|---|
| **Non-determinism** | Same prompt → different response across runs (temperature > 0, model updates, sampling) | Snapshot testing, statistical assertions, fixed seeds, fake/test models |
| **Cost** | Every call to OpenAI/Anthropic costs real money. Running on every PR is expensive | VCR cassettes, marker-based exclusion, test models |
| **Side effects** | Vector store writes, fine-tune jobs, file uploads — hard to roll back | Tmp dirs, in-memory backends, fixture cleanup |
| **External services** | Langfuse / Pinecone / S3 / EDGAR — slow, flaky, rate-limited | Marker isolation, fixture-based skip, recording proxies |
| **Quality not correctness** | "Is this summary *good*?" can't be `assert ==` | Eval-as-test (DeepEval, Ragas), G-Eval, human review |
| **Stateful pipelines** | Feast features depend on prior materializations; agent state carries across nodes | Hermetic per-test stores, state-machine tests |
| **Time + drift** | A test that passed yesterday fails today because GPT-4 was updated | Pinned model versions, model-version-as-test-input |
| **Volume / cardinality** | LLM responses are kilobytes, not booleans — comparing them is expensive | Snapshot tools, structured-output schemas |

The right test strategy treats these as different *tiers* with different cost,
fidelity, and frequency.

---

## 2. The test pyramid stretched for LLM systems

The classic pyramid (many unit, fewer integration, very few E2E) becomes:

```
                                                cost/time
                       ┌──────────────┐         per test
                       │  Eval tests  │           $$$$
                       │ (DeepEval,   │          minutes
                       │  Ragas, G-)  │
                       └──────────────┘
                  ┌──────────────────────┐
                  │ Integration tests    │           $
                  │ (Langfuse, Postgres, │        seconds
                  │ Docker, real LLM)    │
                  └──────────────────────┘
            ┌──────────────────────────────────┐
            │ Property tests (Hypothesis)      │         $
            │  Math invariants, CV folds, ...  │      ms-seconds
            └──────────────────────────────────┘
       ┌──────────────────────────────────────────────┐
       │ Unit tests                                   │         ¢
       │  Pure functions, schemas, error paths        │       <1ms
       └──────────────────────────────────────────────┘
```

How this repo materializes it (after PR #2 merges):

| Tier | Location | Run via | When |
|---|---|---|---|
| Unit | `tests/unit/` | `make test` | every commit (pre-push, CI) |
| Integration | `tests/integration/` | `make integration` | locally before PR; requires Docker |
| Eval (paid API) | `@pytest.mark.eval` | `make eval` | nightly or pre-release; costs $ |

The CI workflow runs only the unit tier. Locally, `make check-full` runs unit
+ integration. Eval is manual.

**Interview answer template:** "We tier tests by cost × determinism. Unit
tier is fast and hermetic — runs in CI. Integration tier needs Docker —
locally only. Eval tier hits paid APIs — gated by a marker and a manual
run. Each tier has its own gate: typecheck + linter for unit; service-up
fixture for integration; cost-cap + dollar threshold for eval."

---

## 3. Mocking discipline (the highest-leverage judgment call)

**The rule:** mock at *system* boundaries, never at *domain* boundaries.

System boundaries (mock these):
- LLM provider SDKs (Anthropic, OpenAI)
- Exchange / brokerage APIs
- Object stores (S3, GCS)
- Time / clock
- File system at the OS edge
- HTTP outbound

Domain boundaries (NEVER mock these):
- Your own coordinator / strategy / risk gate logic
- Your own data structures (`PairIndex`, `BacktestReport`, `ResearchState`)
- Your own pure functions (`triple_barrier_label`, `cpcv_splits`)

The test:

```python
# ✗ WRONG — mocking your own domain
def test_research_decides_promote():
    mock_report = MagicMock()
    mock_report.deflated_sharpe = 1.5
    mock_gate = MagicMock()
    mock_gate.check_t2_gate.return_value = "promote"
    result = decide_node(state, mock_gate)
    assert result.decision == "promote"
# This test verifies the mock was called. The real gate logic is untested.
```

```python
# ✓ CORRECT — mock only the LLM boundary; real gate, real report
def test_research_decides_promote_when_t2_passes():
    report = BacktestReport(
        signal_id="A2",
        deflated_sharpe=1.5,
        sharpe_net=0.9,
        # ... real values
    )
    state = ResearchState(validation_report=report, ...)
    result = decide_node(state)  # real check_t2_gate inside
    assert result.decision == "promote"
```

**Interview signal:** when asked "how do you test the research agent?" —
say "we mock the LLM SDK, never the gate. The gate's `T1_GATE` /
`T2_GATE` constants are code, not configuration; testing them through a
mock would be testing the mock. We run them with real reports
constructed by hand or from fixtures."

This is the #1 distinguisher between "I've written tests" and "I've
thought about what tests are *for*." `BACKTEST.md` §6 and
`AI_CODE_STYLE.md` §6 codify this in this repo.

### When you must mock: prefer fakes over mocks

A *mock* records calls; a *fake* implements behavior. Fakes are cheaper
long-term:

```python
# Mock: every test must specify what `messages.create` returns
mock_client = MagicMock()
mock_client.messages.create.return_value = MagicMock(
    usage=MagicMock(input_tokens=10, output_tokens=20),
    content=[TextBlock(text="ok")],
)

# Fake: a real test client that returns canned but consistent responses
from anthropic import AsyncAnthropic
from anthropic.lib.test import FakeClient  # hypothetical; pydantic-ai has TestModel
fake = FakeClient(responses=["ok", "thinking..."])
```

Pydantic AI ships a `TestModel`; LangChain has `FakeListLLM`. Use them.

---

## 4. Property-based testing with Hypothesis (the math defender)

For algorithms with provable invariants — CPCV folds, triple-barrier
labels, deflated Sharpe — example-based tests can only check the cases
you thought of. Property tests generate inputs:

```python
from hypothesis import given, strategies as st

@given(
    sample_times=st.lists(st.datetimes(), min_size=10, max_size=100),
    n_splits=st.integers(min_value=3, max_value=10),
    n_test_splits=st.integers(min_value=1, max_value=3),
    embargo_pct=st.floats(min_value=0.0, max_value=0.05),
)
def test_cpcv_no_train_test_overlap(sample_times, n_splits, n_test_splits, embargo_pct):
    """Every CPCV split: train and test index sets are disjoint after embargo."""
    if n_test_splits >= n_splits:
        return  # invalid; Hypothesis will try again
    splits = cpcv_splits(
        sample_times=pd.Series(sample_times),
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        embargo_pct=embargo_pct,
    )
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
```

Hypothesis will:
- Try 100 random inputs (default) — flag any failure
- *Shrink* the failing input to a minimal reproduction
- Save the failing seed so the next run re-tests it

When to reach for property testing:
- Math invariants (additivity, monotonicity, idempotency)
- State machine transitions (every state reachable; no orphan states)
- Parsers (round-trip: `parse(serialize(x)) == x`)
- Sort/dedupe/CV splits (set-theoretic invariants)

When NOT to:
- Business logic with many ad-hoc cases (table-driven tests are clearer)
- I/O-heavy code (slow generation, low signal)

**Interview gold:** "We use Hypothesis for CPCV fold validation — the
property is *train and test sets are disjoint after embargo, for every
combination*. Hand-written examples would cover the obvious cases; the
embargo-edge cases come from Hypothesis shrinking a 100-sample input
down to the 3-sample minimum that violates the invariant."

In this repo: planned for Issues #7 (PIT timestamp invariant), #23
(info_tests), #24 (CPCV, labels, deflated Sharpe).

---

## 5. VCR cassettes — testing LLM/HTTP calls without paying

The pattern: record HTTP interactions once into a "cassette" file, then
replay deterministically forever.

```python
import pytest

@pytest.mark.vcr(cassette_library_dir="tests/fixtures/cassettes/edgar")
def test_edgar_fetches_a_10k():
    """Verified against EDGAR once on 2026-05-22; replays deterministically."""
    client = EdgarClient(user_agent="test")
    doc = client.fetch_10k(cik="0000320193", year=2025)
    assert doc.form_type == "10-K"
    assert b"Apple" in doc.content[:1000]
```

First run: hits EDGAR, saves `tests/fixtures/cassettes/edgar/test_edgar_fetches_a_10k.yaml`.
Subsequent runs: replay from the cassette. Commit the cassette to git.

**Why this matters for LLM work:**
- Anthropic / OpenAI calls cost real money — VCR records once, replays free
- Cassettes are deterministic — flake-free CI
- Reviewers see the exact API shape committed alongside the test

**When to re-record:**
- API contract changes (different response shape)
- Schema migration

`pytest-vcr` is the Python lib. `pytest-recording` is the modern
successor. For Anthropic specifically, the `tests/fixtures/cassettes/`
directory will hold dozens of cassettes by W2.

**Interview answer template:** "We use VCR cassettes for any test
hitting an external API — Anthropic, EDGAR, FMP. Record once, replay
forever. The cassette is checked in; when the API changes, the cassette
re-recording is a deliberate step in the PR. Cost: zero per CI run.
Deterministic: yes. Reviewable: yes."

---

## 6. Snapshot testing (syrupy)

For outputs that are hard to assert by hand — generated reports, agent
memos, signal cards, formatted tables:

```python
import pytest
from syrupy import SnapshotAssertion

def test_backtest_report_renders(snapshot: SnapshotAssertion):
    report = BacktestReport(signal_id="A2", deflated_sharpe=1.23, ...)
    assert report.render_markdown() == snapshot
```

First run: writes `__snapshots__/test_xxx.ambr` with the actual output.
Subsequent runs: compares.

When the output legitimately changes:
```bash
pytest --snapshot-update
```

Reviewer sees the snapshot diff in the PR — every change is visible and
intentional. Good for:
- `BacktestReport` and `InfoReport` formatted outputs
- Research-agent memos (markdown)
- Signal cards
- CLI help text

Bad for:
- Non-deterministic outputs (LLM responses without fixed seeds — use
  semantic eval instead)
- Outputs that change every run (timestamps, UUIDs) — sanitize before
  snapshotting

This repo will adopt syrupy in Issues #25 (BacktestReport), #28
(memo rendering), #30 (signal cards).

---

## 7. Mutation testing — testing the tests

The question: "are my tests catching real bugs?" Coverage % can't answer
that (you can have 100% coverage and zero assertions). Mutation testing
can.

**How it works:** a mutation tool (`mutmut`, `cosmic-ray`) mutates the
production code (`>` → `<`, `+` → `-`, `True` → `False`, etc.), then
runs the test suite. If any test fails → mutation *killed*. If all
tests pass → mutation *survived*, meaning your tests didn't catch a
real change in behavior.

```bash
mutmut run --paths-to-mutate=src/auto_research/backtest/cpcv.py
mutmut results
# 87 mutants killed / 93 survived / 12 timeouts → 53% kill rate
```

A high mutation-kill score on `cpcv.py` / `deflated_sharpe.py` /
`labels.py` is the *real* answer to "do you trust your tests."

**When to use it:**
- Sensitive paths (this repo: §3 sensitive paths in `ARCHITECTURE.md`).
- Algorithms with provable invariants — mutations often reveal missing
  edge-case assertions.

**When NOT to use it:**
- Glue code (CLI parsing, logging) — kill rate naturally low; not
  worth optimizing.
- UI rendering — too many mutations are visually-equivalent.

**Cost:** slow (one test-suite run per mutant; ~1000s of mutants for a
module). Run nightly, not per-commit.

**Floor target:** `BACKTEST.md` §6 says ≥85% kill rate on §4 modules.

**Interview gold:** "Coverage isn't our gate; mutation kill-rate on
sensitive paths is. We tolerate low coverage on logging glue but
require ≥85% mutation kill on `cpcv.py` and `deflated_sharpe.py`. The
gate runs nightly because mutation runs are expensive."

This separates senior from mid in interviews.

---

## 8. Fixture scopes and composition

pytest fixtures have four scopes:

| Scope | Lifetime | Use for |
|---|---|---|
| `function` (default) | Recreated per test | State that tests shouldn't share |
| `class` | Per test class | Class-level shared state (rare in pytest-style) |
| `module` | Per test file | Module-shared state |
| `session` | Per pytest session | Expensive setup (Docker, big embeddings, DB) |

This repo's `tests/integration/conftest.py` uses `pytest_collection_modifyitems`
(session-level via hook) rather than `session` fixture because the
Langfuse probe runs once per session, regardless of how many tests.

Example combining scopes:

```python
@pytest.fixture(scope="session")
def langfuse_client():
    """Created once per session; reused across all integration tests."""
    init_telemetry()
    from langfuse import Langfuse
    yield Langfuse()
    # teardown if needed

@pytest.fixture(scope="function")
def fresh_state():
    """A new ResearchState per test — no leakage."""
    return ResearchState(session_id=f"test-{uuid4()}")

def test_agent_emits_memo(langfuse_client, fresh_state):
    # langfuse_client is shared; fresh_state is per-test
    ...
```

**Interview signal:** when asked "how do you handle expensive setup,"
the answer is *not* "we use a global variable" — it's "session-scoped
fixture with teardown via `yield`." Knowing the four scopes by heart
signals depth.

### `monkeypatch` vs `mock.patch`

```python
# monkeypatch — reverts automatically at test teardown
def test_thing(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    monkeypatch.setattr("module.CONSTANT", 42)
    # at function exit, env + attr revert

# mock.patch — explicit context manager or decorator
from unittest.mock import patch

def test_thing():
    with patch("module.thing", return_value=42):
        ...
```

`monkeypatch` is safer (auto-revert). `mock.patch` is more powerful
(can patch chains, supports `spec=` for type-safety). Use monkeypatch
for env + module attributes; use mock.patch for object methods needing
chained `.return_value` setup.

---

## 9. Type-check the tests too

Default mypy setups skip `tests/`. Add it:

```toml
[tool.mypy]
files = ["src/auto_research", "tests"]
strict = true
```

This catches stupid bugs: missing fixture parameters, wrong arg
positions, response objects you forgot to narrow.

This repo just caught a real bug from this exact pattern — `response.content[0]`
is a union of 13 block types; only `TextBlock` has `.text`. mypy on
tests required:

```python
from anthropic.types import TextBlock

first_block = response.content[0]
assert isinstance(first_block, TextBlock)  # narrows type for mypy
assert first_block.text.strip().lower().startswith("ok")
```

Without strict mypy, the test would have run fine on a `TextBlock` (the
expected case) and crashed at runtime if the API ever returned a different
block type. Strict mypy on tests catches this at lint-time.

**Interview gold:** "We mypy the tests too, strict mode. It caught a
real bug — Anthropic's `response.content` is a union and we were
dereferencing `.text` without narrowing. The test would have appeared
to pass until the API returned a non-text block."

---

## 10. Pre-commit hooks (mechanical hygiene before commit)

Pre-commit catches the cheap-to-detect mistakes:

- Trailing whitespace (annoying in diffs)
- Files without trailing newlines (POSIX nag)
- Mixed line endings (Windows-vs-Unix diff noise)
- Unresolved merge conflict markers
- Large files committed by accident (data, binaries)
- Private keys / API tokens
- Invalid YAML / TOML / JSON
- Ruff lint failures

This repo's `.pre-commit-config.yaml` runs these via the `pre-commit`
framework. Install once: `uv run pre-commit install`.

**What to NOT put in pre-commit:**
- `mypy` — slow (~10s cold), runs over too many files
- `pytest` — even slower; CI's job
- Anything that takes >2s — kills the dev loop

Rule of thumb: pre-commit is the cheap mechanical gate. Slow gates
belong in `make check` (developer-invoked) and CI (automatic).

### Custom hooks for project invariants

The `detect-private-key` hook ships with pre-commit but only matches
PEM/OpenSSH formats — useless for LLM API keys. This repo adds a local
pygrep hook:

```yaml
- repo: local
  hooks:
    - id: detect-llm-api-keys
      name: detect LLM API keys (anthropic / langfuse)
      entry: '(sk-ant-[A-Za-z0-9_-]{20,}|pk-lf-[a-f0-9-]+|sk-lf-[a-f0-9-]+)'
      language: pygrep
      types: [text]
      exclude: ^(\.env\.example|docs/.*\.md|tests/.*)$
```

**Interview signal:** "We use pre-commit for the cheap mechanical
checks — whitespace, file format, merge markers, large files, our LLM
API key prefixes. Mypy and pytest run in `make check` and CI because
they're too slow for every commit."

---

## 11. Test discipline grab-bag

A few more patterns interviewers probe for:

### `tmp_path` for file isolation

Never write to `/tmp/foo` from a test. pytest's `tmp_path` fixture
gives you an auto-cleaned-up directory:

```python
def test_writes_quarantine_record(tmp_path):
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    extract_with_quarantine(quarantine_dir=quarantine)
    assert (quarantine / "doc.json").exists()
```

### Pytest markers + `--strict-markers`

```toml
[tool.pytest.ini_options]
addopts = "--strict-markers"
markers = [
    "eval: tests that hit paid LLM/eval APIs",
    "integration: tests that hit external services",
]
```

Without `--strict-markers`, a typo like `@pytest.mark.intregration`
silently runs in every selection. With it, pytest errors loudly.

### Test naming convention

Pick one and apply consistently:

- `test_<unit>_<scenario>_<expected>` — `test_init_telemetry_raises_when_env_missing`
- `test_when_<state>_then_<outcome>` — BDD-style

This repo uses the first.

### Parameterize for table-driven tests

```python
@pytest.mark.parametrize(
    "n_splits, n_test_splits, expected_combinations",
    [
        (3, 1, 3),     # C(3,1) = 3
        (4, 2, 6),     # C(4,2) = 6
        (6, 2, 15),    # C(6,2) = 15
    ],
)
def test_cpcv_yields_correct_combinations(n_splits, n_test_splits, expected_combinations):
    splits = list(cpcv_splits(n_splits=n_splits, n_test_splits=n_test_splits, ...))
    assert len(splits) == expected_combinations
```

One test function, many cases. Cleaner than three near-identical functions.

### Async testing

```python
import pytest

@pytest.mark.asyncio
async def test_async_llm_call():
    client = AsyncAnthropic()
    response = await client.messages.create(...)
    assert response.usage.input_tokens > 0
```

Requires `pytest-asyncio`. Configure mode in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### CI parallelization (`pytest-xdist`)

`pytest -n auto` runs the suite across available cores. Speed wins on
slow suites. Requires test isolation (no shared global state). Worth
adding once the unit suite passes 100 tests.

---

## 12. Anti-patterns

| Anti-pattern | Why it's wrong |
|---|---|
| `MagicMock()` on your own domain objects | Tests verify the mock was called, not real behavior |
| Sharing test state via module-level vars | Order-dependent failures, hard to debug |
| Skipping flaky tests instead of fixing them | "Flaky" is usually "racy" — investigate |
| Coverage % as the only gate | High coverage with weak assertions = false confidence |
| `time.sleep(N)` waiting for async work | Flake-prone; use polling with timeout or event-based wait |
| One huge test that exercises 12 things | When it fails, you don't know which thing broke |
| Tests that depend on PWD or local time | Use `tmp_path`, freeze time with `freezegun` |
| Hard-coded API keys in test fixtures | Leak into git history; use env vars + skip |
| Mocking `time.time()` everywhere | Use `clock` injection or `freezegun` for the few tests that need it |
| Ignoring the test pyramid (only integration tests) | Slow CI; tests don't isolate failures |

---

## 13. Interview Q&A with answer templates

### Q: "How do you test something stochastic like an LLM call?"

> Three approaches, ordered by determinism. (1) Pin the model version + use temperature=0 + seed when supported — gets you near-determinism for many models. (2) VCR cassettes — record real responses once, replay forever; deterministic but not regenerative. (3) Semantic eval (DeepEval G-Eval, Ragas faithfulness) — accept the output varies, judge it on quality. Choose by what you're testing: contract → VCR; quality → eval; rendering → snapshot.

### Q: "What do you mock and what do you NOT mock?"

> Mock at system boundaries — LLM SDKs, exchange APIs, time, file system, network. Never mock at domain boundaries — our own coordinator, gate, schemas, pure functions. The test for the distinction: if mocking it means I'm testing the mock instead of the behavior, I'm wrong. Concrete example: `T2_GATE` is a code-level constants dict; testing the `decide` node with a mocked gate would assert the mock returned what we told it to. Testing with a real gate + a hand-constructed `BacktestReport` actually exercises the threshold logic.

### Q: "Why not 100% code coverage?"

> Coverage measures line execution, not behavior. A test that imports a module and asserts nothing achieves 100% coverage and proves nothing. We track coverage as a sanity check but gate on mutation kill-rate for sensitive modules — ≥85% kill rate on `cpcv.py` / `deflated_sharpe.py`. Mutation testing is the actual answer to "do these tests catch bugs?" Coverage alone is theater.

### Q: "How do you test code that costs money to run?"

> Three layers. (1) Marker-based exclusion — `@pytest.mark.eval` tests don't run in CI by default. (2) VCR cassettes for HTTP calls — record once locally, replay free in CI. (3) Cost-cap decorators on the agent itself — each test session has a hard $ ceiling, exceeding it raises. The combination means CI runs are free; local pre-PR runs hit cassettes; eval runs are intentional + tracked.

### Q: "Property-based testing — when?"

> When the function has an *invariant* easier to state than its full input space. CPCV: "train and test sets are disjoint after embargo." Triple-barrier label: "label is in {-1, 0, +1} for any vol-adjusted band." Serializer round-trip: `parse(serialize(x)) == x`. Hypothesis generates random inputs and shrinks failing cases to a minimum reproduction. We use it for the math-heavy CPCV / deflated Sharpe / labels modules. We don't use it for business logic with many ad-hoc cases — table-driven tests are clearer there.

### Q: "Tell me about a test that caught a real bug."

> Type-checking the test suite (strict mypy on `tests/` too) caught a bug in our integration smoke. We were accessing `response.content[0].text` from an Anthropic call — fine in practice, but `response.content[0]` is a union of 13 block types; only `TextBlock` has `.text`. mypy required an `isinstance(TextBlock)` narrowing. Without it, the test would have crashed at runtime the moment the API returned a thinking block or tool-use block. The fix took 30 seconds; finding it manually would have taken a confused debugging session in production.

### Q: "How do you organize tests in a Python project?"

> `tests/unit/` for hermetic, fast, no-network tests — runs in CI on every PR. `tests/integration/` for tests needing external services (Langfuse, Postgres) — runs locally. Per-folder `conftest.py` for setup specific to that tier. Auto-apply markers via `pytest_collection_modifyitems` — but path-filter on the items list because the hook receives the session-wide collection. Skip-by-fixture for service-up checks. Makefile targets that match the tiers: `make test`, `make integration`, `make eval`, `make check`, `make check-full`.

---

## 14. Patterns concrete to this repo

A walk through choices you can defend:

### `tests/integration/conftest.py` auto-applies the `integration` marker

```python
def pytest_collection_modifyitems(config, items):
    integration_items = [item for item in items if "tests/integration" in str(item.path).replace("\\", "/")]
    if not integration_items:
        return

    for item in integration_items:
        item.add_marker(pytest.mark.integration)

    if not _langfuse_healthy():
        skip = pytest.mark.skip(reason="Langfuse not reachable on :3000...")
        for item in integration_items:
            item.add_marker(skip)
```

Three deliberate choices:

1. **Path filter on `items`** — the hook receives the *session-wide* list,
   not a subtree. Without the filter, unit tests collected in the same
   run get stamped with `integration` too. (This repo's first attempt
   got bit by exactly that — caught in code review.)
2. **Health-endpoint probe** instead of raw socket — a stray service on
   :3000 doesn't fool us; we hit `/api/public/health` and trust
   `urlopen`'s 4xx/5xx raise behavior.
3. **Skip via marker** rather than `pytest.skip()` in a fixture — same
   effect, but the test report shows "skipped" cleanly with the reason
   visible.

### `make test` / `make integration` / `make eval` separation

Different prerequisites, different cost. CI runs `make test` only.
`make eval` requires API keys and bills the configured tier. The split
exists so a developer knows what each invocation costs.

### `pyproject.toml` mypy on tests with `langfuse.*` override

```toml
[[tool.mypy.overrides]]
module = "langfuse.*"
ignore_missing_imports = true
```

Langfuse v2 doesn't ship type stubs (no `py.typed`). Rather than mypy
fail on every import, we override. The override is module-specific,
not a blanket `ignore_errors` — strict mode applies everywhere else.
Each entry in this overrides list represents a *known* untyped surface.

### Pre-commit's custom LLM-API-key hook

The bundled `detect-private-key` hook only matches PEM/OpenSSH formats.
We added a local pygrep hook for the prefixes we know:
`sk-ant-`, `pk-lf-`, `sk-lf-`. The hook excludes `.env.example`,
`docs/*.md`, and `tests/*` to avoid false-positives on documentation
strings.

This is a small thing but it's the kind of detail that signals
project-specific thoughtfulness to a reviewer.

---

## 15. Suggested learn-order

1. **Read pytest docs end-to-end.** Yes, all of it. ~4 hours. The doc is
   well-written; you'll learn things you didn't know you didn't know.
   Particularly fixture scopes, parameterize, conftest hierarchy.
2. **Read `tests/integration/conftest.py` in this repo.** ~10 min. Connect
   the docs to a real implementation that survived a P0 bug fix.
3. **Try Hypothesis on a function you wrote.** ~30 min. Pick any function
   with a clear invariant. Run it. Watch Hypothesis shrink a failing
   input.
4. **Set up `pytest-vcr` on an HTTP-calling function.** ~30 min. Record
   once, commit the cassette, re-run with the network off. Magic.
5. **Run `mutmut` on a small algorithm module.** ~30 min plus the slow
   mutation run. The first time you see surviving mutants in code you
   thought was well-tested is humbling.
6. **Read about test categorization in real OSS projects.** Pick three
   well-tested Python projects (pandas, requests, FastAPI). Look at
   their `tests/` layout, conftest hierarchy, marker usage. ~1 hour.
7. **Pick one anti-pattern from §12** that's in your past code and rewrite
   it. ~30 min.

Total: ~8 hours to be genuinely fluent.

After that, "how do you test X" stops being a memorization question and
becomes a reasoning exercise. The interview signal is your ability to
articulate *why* you chose the test you wrote — not whether you know
the API surface of pytest.
