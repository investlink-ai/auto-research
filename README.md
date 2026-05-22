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
uv sync --all-extras
make check
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the worktree convention,
branch + commit format, and per-issue planning workflow.
