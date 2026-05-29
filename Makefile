.PHONY: quick check check-full test test-broad integration eval live-smoke lint typecheck setup-nlp setup-mlx setup-reranker smoke serve-local-llm

# Fast pre-commit gate — run constantly during development.
quick: lint typecheck

# Default pre-PR gate: lint + typecheck + unit tests. Cheap, hermetic.
check: quick test

# One-time NLP setup. Pre-downloads two lazy-loaded models that would
# otherwise hit the network on first test/use:
#
#   1. spaCy `en_core_web_sm` — needed by
#      `unstructured.partition_html`'s element classifier. Hosted only
#      on GitHub (not on PyPI), so `uv sync` does not install it.
#   2. BGE `BAAI/bge-small-en-v1.5` — the in-process embedding fallback
#      used by `EmbeddingAdapter` when `VOYAGE_API_KEY` is absent.
#      `sentence-transformers` lazy-loads from HuggingFace on first
#      instantiation; pre-warming the HF cache here lets the
#      conftest's session-autouse fixture serve hermetic tests without
#      network access.
#
# Both warmups are idempotent — re-running is a no-op once the caches
# are populated. CI runs this in the same step as `uv sync`.
setup-nlp:
	uv run python -m spacy download en_core_web_sm
	uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Apple-Silicon-only Qwen3-Embedding MLX warmup. Pre-pulls the dev
# default (0.6B, ~600 MB) into the HuggingFace cache so the
# conftest's session-autouse fixture serves hermetic Mac unit tests
# without network access. The 4B variant (~8 GB) is opt-in via
# `QWEN3_FULL=1` — it's deployment-grade weight, not a default
# developer-machine download. Idempotent; re-running with a populated
# HF cache is a no-op. Skips gracefully on non-Apple-Silicon hosts
# (the `mlx-embeddings` extra isn't installed there).
setup-mlx:
	@if [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
		uv run --extra mlx python -c "from auto_research.extract.embeddings import _ensure_qwen3_warmup; _ensure_qwen3_warmup('Qwen3-Embedding-0.6B')"; \
		if [ "$$QWEN3_FULL" = "1" ]; then \
			uv run --extra mlx python -c "from auto_research.extract.embeddings import _ensure_qwen3_warmup; _ensure_qwen3_warmup('Qwen3-Embedding-4B')"; \
		fi; \
	else \
		echo "setup-mlx: skipped (non-Apple-Silicon host); MLX backend is Mac-only."; \
	fi

# Cross-platform Qwen3-Reranker warmup. Pre-pulls the 0.6B weights
# (~1.2 GB) into the HuggingFace cache so the conftest's session-autouse
# fixture serves hermetic tests without network access. Loads on CPU
# regardless of host (the cache entry is keyed by device, so the MPS
# entry on Apple Silicon is still lazily populated by the first test
# that exercises it). The 4B variant (~8 GB) is opt-in via
# `QWEN3_FULL=1`. Idempotent.
setup-reranker:
	uv run python -c "from auto_research.extract.rerank import _ensure_qwen3_reranker_warmup; _ensure_qwen3_reranker_warmup('Qwen3-Reranker-0.6B', 'cpu', 'fp32')"
	@if [ "$$QWEN3_FULL" = "1" ]; then \
		uv run python -c "from auto_research.extract.rerank import _ensure_qwen3_reranker_warmup; _ensure_qwen3_reranker_warmup('Qwen3-Reranker-4B', 'cpu', 'fp32')"; \
	fi

# Full local gate: + integration. Requires `docker compose up -d` first.
check-full: quick test integration

# Spin up the locked local-LLM serving stack
# (vllm-mlx + Qwen3.6-35B-A3B UD-MLX-4bit on Mac). Thin wrapper around
# `scripts/serve_local_llm.sh` — that script is the source of truth for
# the launch flags. Foreground; Ctrl-C stops the server. Override
# defaults via env vars: MODEL=..., PORT=..., VLLM_MLX_VENV=...
serve-local-llm:
	./scripts/serve_local_llm.sh

lint:
	uv run ruff check .

typecheck:
	uv run mypy

# Unit tests — hermetic, no network, no Docker, no API keys.
# tests/feast is included here: it scaffolds a tmp Feast registry from local
# files and runs `feast apply` in-process; no network.
#
# Chunking fixtures are tiered: `core` fixtures run by default (small,
# intentional set of templates), `broad` fixtures cover wider industry/
# year/template variance and run only via `make test-broad`. Each
# chunking fixture has a `tier` field in its meta.json.
test:
	uv run pytest tests/unit tests/feast -m "not broad_fixture"

# Same as `make test` plus the broad-tier chunking fixtures. MANUAL —
# not run in CI (default or nightly). Touching `src/auto_research/
# extract/chunking.py` (Tier 2 per AGENTS.md §3) requires running this
# locally before merge and citing the green output in the PR evidence
# block. This is the regression-coverage backstop for filer-template
# variance the core set doesn't reach.
test-broad:
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
