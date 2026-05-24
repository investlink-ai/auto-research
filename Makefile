.PHONY: quick check check-full test integration eval live-smoke lint typecheck smoke

# Fast pre-commit gate — run constantly during development.
quick: lint typecheck

# Default pre-PR gate: lint + typecheck + unit tests. Cheap, hermetic.
check: quick test

# Full local gate: + integration. Requires `docker compose up -d` first.
check-full: quick test integration

lint:
	uv run ruff check .

typecheck:
	uv run mypy

# Unit tests — hermetic, no network, no Docker, no API keys.
# tests/feast is included here: it scaffolds a tmp Feast registry from local
# files and runs `feast apply` in-process; no network.
test:
	uv run pytest tests/unit tests/feast

# Integration tests — require Langfuse running (docker compose up -d).
# Tests skip cleanly if Langfuse isn't reachable on :3000.
integration:
	uv run pytest tests/integration

# Paid-API suites (DeepEval, Ragas, anything marked `eval`).
# Excluded from CI by default; runs locally and bills the configured API keys.
eval:
	uv run pytest -m eval

# Live-smoke suites — hit real SEC / FMP / Anthropic endpoints.
# Excluded from per-PR CI; runs nightly via .github/workflows/live-smoke.yml
# and locally on-demand. Tests skip cleanly when their required env vars
# (declared per-module via `live_requires_env`) aren't set.
live-smoke:
	uv run pytest tests/live -m live

# W1 acceptance smoke - one ticker, one S-3, end-to-end.
# Requires SEC_USER_AGENT + ANTHROPIC_API_KEY in env.
# Idempotent against the manifest + extract cache - re-runs are no-ops.
# Default ticker is NVDA (CIK 0001045810); override with SMOKE_CIK=...
SMOKE_CIK ?= 0001045810
smoke:
	uv run auto-research ingest edgar --cik $(SMOKE_CIK) --form-types S-3
	uv run auto-research extract s-filings --cik $(SMOKE_CIK)
	uv run auto-research feast apply
	uv run auto-research status
