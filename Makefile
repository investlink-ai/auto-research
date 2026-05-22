.PHONY: quick check test lint typecheck eval

# Fast pre-commit gate — run constantly during development.
quick: lint typecheck

# Pre-PR gate — quick + unit tests (excludes paid-API and integration tests).
check: quick test

lint:
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not eval and not integration"

# Paid-API + integration suites. Run locally; excluded from CI by default.
eval:
	uv run pytest -m "eval or integration"
