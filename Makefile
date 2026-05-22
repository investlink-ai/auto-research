.PHONY: quick check check-full test integration eval lint typecheck

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
