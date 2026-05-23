.PHONY: quick check check-full test integration eval live-smoke lint typecheck

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
test:
	uv run pytest tests/unit

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
