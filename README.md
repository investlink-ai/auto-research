# auto-research

Two-plane multi-agent research platform for cross-asset language-driven
alpha in AI infrastructure and frontier-tech equities — the engineering
corrective to `virattt/ai-hedge-fund`'s LLM-at-wrong-layer anti-pattern.

The LLM never sits in the trading hot path. Extraction is nightly batch
with content-hash caching; research is asynchronous; the live critic
emits a multiplicative haircut, never a directional override.

## Where to read

- [`docs/specs/2026-05-22-design.md`](docs/specs/2026-05-22-design.md) — frozen v1 design + rationale + interview narrative.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current module map + plane boundaries.
- [`docs/CONTRACTS.md`](docs/CONTRACTS.md) — Pydantic schemas + MCP tool surface.
- [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) — Feast FeatureViews + PIT contract.
- [`docs/BACKTEST.md`](docs/BACKTEST.md) — T1/T2 gates + CPCV + deflated Sharpe.
- [`AGENTS.md`](AGENTS.md) — invariants + working style for AI agents and humans.
- [`docs/plans/2026-05-22-auto-research-implementation.md`](docs/plans/2026-05-22-auto-research-implementation.md) — 32-issue implementation plan.

## Status

W1 — Foundation. Build target: 4 weeks from 2026-05-22.

Track progress in the [issue tracker](../../issues) and
[milestones](../../milestones). Per-signal results land under
`docs/signal_cards/` as they survive the T2 backtest gate.

## Quickstart

```bash
# 1. Python project + tooling
uv sync --all-extras
uv run pre-commit install     # mechanical checks on every commit
make check                    # ruff + mypy + unit tests

# 2. Langfuse self-hosted for LLM traces (one-time per machine)
docker compose up -d
# open http://localhost:3000, sign up, create a project, copy keys → .env

# 3. Verify telemetry end-to-end (requires Docker running + .env populated)
make integration

# 4. Browse experiment tracker (after backtests have run)
uv run mlflow ui              # opens http://localhost:5000
```

## Test layers

| Target | Scope | Cost / prerequisites |
|---|---|---|
| `make test` | `tests/unit/` — hermetic | none |
| `make integration` | `tests/integration/` — needs Langfuse running | `docker compose up -d` |
| `make eval` | `@pytest.mark.eval` — paid API tests (DeepEval, Ragas) | `ANTHROPIC_API_KEY` + $ |
| `make check` | `quick` + `test` | none (default pre-PR gate) |
| `make check-full` | `quick` + `test` + `integration` | Langfuse running |

## Prompt lifecycle

Prompts live as `<NAME>_PROMPT` constants in
`src/auto_research/extract/prompts/`, colocated with
`<NAME>_PROMPT_VERSION = "vN"`. Code is the source of truth at runtime;
Langfuse holds the registry for version history and tag state. The
discipline:

1. **Edit** a prompt file → the `bump-prompt-version` skill blocks the
   commit unless `*_PROMPT_VERSION` also bumps. If the partnered
   Pydantic output model's fields changed, the skill also requires
   bumping the model's `SCHEMA_VERSION` ClassVar.
2. **Cache** is keyed on the full completion config
   (`raw_doc`, `prompt_version`, `schema_version`, `model_id`,
   `decoding_params`) — see `src/auto_research/extract/cache.py`. Model
   swaps and decoding changes never reuse stale entries.
3. **Promote** to the Langfuse `production` tag via
   `uv run python scripts/promote_prompt.py <prompt_name> <version>` —
   the script runs the gold-set eval and refuses the tag flip below the
   F1 threshold or above the per-doc cost ceiling.

See `AGENTS.md` INV-6 for the formal invariant.

## Workflow

[`AGENTS.md`](AGENTS.md) is the source of truth for both human and
AI-agent workflow — invariants, sensitive paths, working style.
[`docs/AI_WORKFLOW.md`](docs/AI_WORKFLOW.md) covers the issue → worktree
→ PR loop, tier classification, plan lifecycle, and verification gates.
