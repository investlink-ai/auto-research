"""OpenAI-compatible HTTP wrapper for extraction workers (local serving).

A second backend behind the same `ExtractionFn` Protocol that
`make_extraction_client` (Anthropic SDK) implements. The motivating
deployment shape: an OpenAI-compatible HTTP server hosted on the same
machine as the rest of the extraction pipeline — Ollama, vLLM,
MLX-server, or llama.cpp all expose the OpenAI v1 chat-completions
shape on `:11434` / `:8000`. One wrapper covers all of them; selecting
which is a `base_url` config decision, not a code decision.

Why this exists at all:
`learning/2026-05-28-extraction-pipeline-cost-model.md` §9 + §10 walk
through the cost case (~$430/yr Anthropic Haiku spend → ~$12/yr on a
local Qwen 3.5 path) and the quality cleavage (templated extraction
goes local; subjective judgment stays on API). This module is Phase 1
of the rollout — infra-only. Per the issue's acceptance criteria, no
routes flip to `local/*` in this PR; that happens per-worker as eval
validates the substitution.

What the wrapper deliberately doesn't do:

- **Cost cap** — local serving has no per-call $$. The hard cap
  primitive (`@cost_cap`) wraps a `Message`-returning Anthropic call;
  it doesn't apply here. The right backpressure primitive for local
  is queue-depth limiting (vLLM's `--max-num-seqs`) or rate-limiting
  on the wrapper; both are follow-ups.
- **Extended thinking** — most OSS models don't expose a separable
  thinking channel; routing to local already means the task is
  Haiku-tier templated work that doesn't benefit from thinking.
- **Prompt caching** — vLLM has a prefix cache but exposes hit/miss
  via different telemetry (`/metrics`, not per-response token counts).
  Adopting it is a deployment configuration concern, not a wrapper
  concern; the `cache_*` fields in `UsageDict` stay absent for this
  backend.

The reliability composition mirrors `client.py`'s Anthropic path but
swaps the cost-cap layer for nothing (so the chain is
`circuit_breaker(retry_with_backoff(...))` only) and uses an
OpenAI-aware retry classifier instead of the Anthropic predicate
hard-coded in `reliability._is_retryable`.

Production callers compose:

    _CLIENT = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    text, usage = _CLIENT(
        task="contextual_chunk",
        system_prompt=PROMPT,
        user_content=child.text,
    )
"""

from __future__ import annotations

import json
import threading
from typing import Any, Final, cast

import openai
from openai.types.chat import ChatCompletion
from opentelemetry import trace
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from auto_research._models import route_model
from auto_research._transport import TRANSIENT_NETWORK_ERRORS
from auto_research.agents.reliability import circuit_breaker
from auto_research.extract._response import UsageDict
from auto_research.extract.client import (
    RECORD_EXTRACTION_TOOL_NAME,
    ExtractionFn,
)

# Default Ollama OpenAI-compat endpoint. Used when no `base_url` is
# passed and no `LOCAL_INFERENCE_URL` env var is set. A pydantic-settings
# consolidation of the local-inference URL lands as a follow-up; for
# now the explicit kwarg wins, falling back to env, falling back to
# this constant. Naming `OLLAMA_DEFAULT_BASE_URL` instead of e.g.
# `LOCAL_DEFAULT_BASE_URL` because vLLM / MLX-server default to other
# ports (`:8000`); the Ollama default is the one we document as the
# path of least resistance.
OLLAMA_DEFAULT_BASE_URL: Final[str] = "http://localhost:11434/v1"

# Sentinel API key passed to `openai.OpenAI()` when the local server
# doesn't require one (Ollama out of the box). The SDK rejects empty
# strings at construction time; literal `"ollama"` keeps the
# constructor happy without implying real secret material.
_LOCAL_API_KEY_SENTINEL: Final[str] = "local"


def _is_openai_retryable(exc: BaseException) -> bool:
    """Tenacity `retry_if_exception` predicate scoped to OpenAI + httpx.

    Mirrors `reliability._is_retryable` but reads OpenAI exception
    classes — 429 (`RateLimitError`) and 5xx `APIStatusError` retry;
    4xx programmer errors (400 bad request, 401 unauthorized,
    404 model-not-found) propagate so they fail loud. `APIConnectionError`
    (raised by the SDK on transport failures it wraps) also retries.
    `TRANSIENT_NETWORK_ERRORS` covers raw httpx errors that bypass the
    SDK's wrapping (rare; kept for symmetry with the Anthropic path).
    """
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return 500 <= exc.status_code < 600
    if isinstance(exc, openai.APIConnectionError):
        return True
    return isinstance(exc, TRANSIENT_NETWORK_ERRORS)


def _completion_to_usage_dict(completion: ChatCompletion) -> UsageDict:
    """Lift OpenAI `CompletionUsage` + `finish_reason` into `UsageDict`.

    OpenAI names tokens `prompt_tokens` / `completion_tokens`; we
    rename to the provider-agnostic `input_tokens` / `output_tokens`
    so callers don't branch on backend. `usage` is `Optional` on the
    SDK type (streaming sets it `None` mid-stream); for sync chat
    completions the field is always present on the final response,
    but we treat `None` as zero rather than raising — local servers
    occasionally omit the usage block on degenerate responses, and
    `0` is the honest fallback.

    `cache_*` fields are deliberately not populated: vLLM's prefix
    cache reports hit-rate via `/metrics`, not per-response. Wrappers
    add it as a `NotRequired` extension once a local backend surfaces
    the signal in the response shape.

    `finish_reason` flows through as-is — OpenAI emits `{stop, length,
    tool_calls, content_filter}`; consumers that care (today: the
    contextual chunker's drop logic) widen their allow-set when they
    flip to the local route.
    """
    usage = completion.usage
    out: UsageDict = {
        "input_tokens": usage.prompt_tokens if usage is not None else 0,
        "output_tokens": usage.completion_tokens if usage is not None else 0,
    }
    # `choices` is required by the OpenAI schema and a degenerate empty
    # list would already have failed schema validation in the SDK; the
    # guard documents the field's presence at this point in the flow.
    if completion.choices:
        finish_reason = completion.choices[0].finish_reason
        if finish_reason is not None:
            out["stop_reason"] = finish_reason
    return out


def _local_model_id(model: str) -> str:
    """Strip the `local/` routing-table prefix to get the server-native ID.

    `route_model("contextual_chunking", "contextual_chunk")` returns
    e.g. `"local/qwen3.5:9b"` (Ollama tag). The OpenAI-compat HTTP API
    expects the bare model name (`"qwen3.5:9b"`) in the request body;
    the `local/` prefix is a dispatch hint for `_get_client`, not a
    real model identifier.
    """
    if model.startswith("local/"):
        return model[len("local/") :]
    return model


def make_openai_compat_extraction_client(
    *,
    worker: str,
    base_url: str | None = None,
    api_key: str | None = None,
    failures: int = 3,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    openai_client: openai.OpenAI | None = None,
) -> ExtractionFn:
    """Build a per-worker extraction callable backed by an OpenAI-compat server.

    Same Protocol shape as `make_extraction_client`. Workers don't see
    which backend they're talking to — the dispatch in
    `workers/_common._get_or_build_client` picks one based on the
    `local/` prefix in the routed model_id.

    Args:
        worker: feeds `route_model(worker, task)`; also tags the
            circuit-breaker state for debugging.
        base_url: OpenAI-compat HTTP endpoint. Defaults to
            `OLLAMA_DEFAULT_BASE_URL`. For vLLM use
            `"http://localhost:8000/v1"`; for MLX-server use the port
            you launched it on. A pydantic-settings consolidation of
            this URL is a follow-up.
        api_key: passed verbatim to `openai.OpenAI`. Local servers
            (Ollama, vLLM) ignore the value but the SDK requires a
            non-empty string at construction. Default `"local"` works
            for both.
        failures: consecutive-failure threshold for `@circuit_breaker`
            (reused unchanged from the Anthropic path's reliability
            primitive).
        max_retries: additional attempts after the first for the
            tenacity-based retry layer (429 / 5xx / connection errors).
        initial_wait, max_wait: exponential-jitter backoff bounds.
        openai_client: optional injected client (for testing). When
            provided, `base_url`/`api_key` are ignored.
    """
    if openai_client is not None:
        sdk = openai_client
    else:
        sdk = openai.OpenAI(
            base_url=base_url if base_url is not None else OLLAMA_DEFAULT_BASE_URL,
            api_key=api_key if api_key is not None else _LOCAL_API_KEY_SENTINEL,
        )

    # Per-client memo for the same reason as the Anthropic wrapper:
    # `model_json_schema()` is not cached by pydantic and the same
    # 4-7 schemas are reused across thousands of calls. Cache by class
    # identity so a re-defined schema (same name, new fields) doesn't
    # return a stale entry.
    _schema_cache: dict[int, dict[str, Any]] = {}

    def _response_format_schema(output_schema: type[BaseModel]) -> dict[str, Any]:
        cached = _schema_cache.get(id(output_schema))
        if cached is None:
            cached = output_schema.model_json_schema()
            _schema_cache[id(output_schema)] = cached
        return cached

    @circuit_breaker(failures=failures)
    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
        retry=retry_if_exception(_is_openai_retryable),
        reraise=True,
    )
    def _call_inner(
        *,
        task: str,
        system_prompt: str,
        user_content: str,
        output_schema: type[BaseModel] | None = None,
        max_tokens: int = 4096,
    ) -> ChatCompletion:
        # Route first so a misspelled `task=` raises before the network.
        # `route_model` raises `ValueError` naming the bad (worker, task)
        # pair; circuit-breaker counts the failure (matching the
        # Anthropic path's behavior) but won't burn the retry budget
        # since `ValueError` isn't in the retry classifier.
        routed = route_model(worker, task)
        model_for_api = _local_model_id(routed)

        # Server-side schema enforcement via OpenAI's structured-outputs
        # `response_format=json_schema`. Functionally parallel to the
        # Anthropic path's forced `tool_choice` — the server constrains
        # decoding to the schema, the response's `message.content` is
        # the JSON-string the wrapper deserializes below. `strict=True`
        # gates whether the server should refuse to decode off-schema
        # tokens; Ollama/vLLM honor it on recent versions but degrade
        # to best-effort on older builds.
        extra_kwargs: dict[str, Any] = {}
        if output_schema is not None:
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": RECORD_EXTRACTION_TOOL_NAME,
                    "schema": _response_format_schema(output_schema),
                    "strict": True,
                },
            }

        # `**extra_kwargs` (carrying `response_format` when an output
        # schema is set) makes the SDK's overload set ambiguous; mypy
        # falls back to `Any`. Cast back to `ChatCompletion` here so
        # downstream type narrowing (`completion.choices`,
        # `completion.usage`) stays meaningful.
        completion = cast(
            ChatCompletion,
            sdk.chat.completions.create(
                model=model_for_api,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                **extra_kwargs,
            ),
        )

        # Local serving has no per-call USD figure, so the Anthropic
        # path's `llm.cost.est_usd` attribute is omitted. Tokens +
        # backend identity ARE useful (lets dashboards split local
        # vs API tier breakdowns), so we set those.
        span = trace.get_current_span()
        span.set_attribute("llm.backend", "openai_compat")
        span.set_attribute("llm.local_model_id", model_for_api)
        if completion.usage is not None:
            span.set_attribute("llm.input_tokens", completion.usage.prompt_tokens)
            span.set_attribute(
                "llm.output_tokens", completion.usage.completion_tokens
            )

        return completion

    def _call(
        *,
        task: str,
        system_prompt: str,
        user_content: str,
        output_schema: type[BaseModel] | None = None,
        max_tokens: int = 4096,
    ) -> tuple[dict[str, Any] | str | None, UsageDict]:
        # The reliability stack wraps `_call_inner` (ChatCompletion
        # return) so circuit-breaker / retry state lives on the inner
        # call; the public callable handles the parse-to-tuple step
        # here.
        completion = _call_inner(
            task=task,
            system_prompt=system_prompt,
            user_content=user_content,
            output_schema=output_schema,
            max_tokens=max_tokens,
        )
        usage = _completion_to_usage_dict(completion)
        content = completion.choices[0].message.content if completion.choices else None

        if output_schema is None:
            # Free-form text path: surface `content` as-is, possibly
            # the empty string if the server emitted no text.
            return content or "", usage

        # Structured path: `response_format=json_schema` puts a JSON
        # string in `content`. `None` content means the server emitted
        # a refusal / content-filter response with no usable structured
        # payload — surface that to the caller as `None` so the
        # quarantine path mirrors the Anthropic "no tool_use block"
        # outcome.
        if content is None:
            return None, usage
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # `strict=True` should make this unreachable on a compliant
            # server, but Ollama/vLLM with older versions degrade to
            # best-effort JSON; an undecodable string is "no usable
            # structured payload" from the caller's perspective, same
            # as `None`.
            return None, usage
        if not isinstance(parsed, dict):
            # Top-level non-dict (`json.loads("123")`, `"[]"`, …) is
            # off-schema for the `response_format=json_schema` contract;
            # treat it as no structured payload rather than passing a
            # weakly-typed value upstream.
            return None, usage
        return parsed, usage

    return _call


# Production singleton table mirrors `workers/_common._CLIENTS`. The
# OpenAI-compat dispatch (`_get_or_build_client` in `_common.py`)
# stores fresh-built clients here keyed by `(worker, model_id)` so
# circuit-breaker state accumulates across calls in production while
# tests injecting `openai_client=` get a fresh per-test client. The
# table is module-level (not factory-closure) so a multi-worker
# backfill that touches both `local/qwen3.5:9b` and `local/qwen3.5:27b`
# keeps separate state per (worker, model) pair without the dispatch
# function having to thread a registry through.
_LOCAL_CLIENTS: dict[tuple[str, str], ExtractionFn] = {}
_LOCAL_CLIENTS_LOCK = threading.Lock()


def get_or_build_local_client(
    worker: str,
    model_id: str,
    *,
    base_url: str | None = None,
) -> ExtractionFn:
    """Return the production singleton for `(worker, model_id)`.

    Tests bypass this by constructing `make_openai_compat_extraction_client`
    directly with `openai_client=...` injected — see
    `tests/unit/test_extract_openai_compat_client.py`.

    `model_id` is the routed value (`local/qwen3.5:9b`) — we key on it
    raw so a future routing flip from 9B to 27B builds a fresh client
    rather than reusing the stale 9B singleton.
    """
    key = (worker, model_id)
    with _LOCAL_CLIENTS_LOCK:
        client = _LOCAL_CLIENTS.get(key)
        if client is None:
            client = make_openai_compat_extraction_client(
                worker=worker,
                base_url=base_url,
            )
            _LOCAL_CLIENTS[key] = client
        return client


def _reset_local_clients_for_testing() -> None:
    """Drop the singleton table — used by the test conftest between cases."""
    with _LOCAL_CLIENTS_LOCK:
        _LOCAL_CLIENTS.clear()


__all__ = [
    "OLLAMA_DEFAULT_BASE_URL",
    "get_or_build_local_client",
    "make_openai_compat_extraction_client",
]
