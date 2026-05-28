"""Anthropic-SDK wrapper for extraction workers (Issue #10).

A thin layer on top of `anthropic.Anthropic` that bakes in the four
things workers shouldn't have to remember per-call:

1. **Tiered model routing** — `route_model(worker, task)` picks the
   right Claude tier per `docs/specs/2026-05-22-design.md` §7.3. No
   `model="claude-..."` literals scattered across worker files.
2. **Prompt caching policy** — the system prompt is always marked
   `cache_control: {"type": "ephemeral"}`. Spec §7.4's "~80% cached at
   \\$0.30/M" economics depend on the long stable prefix (the system
   prompt) being cached and per-doc user content being fresh. Workers
   needing different breakpoints should call the SDK directly.
3. **Reliability composition** — `@reliable_agent_node(failures, usd,
   max_retries)` wraps every call so cost-cap, circuit-breaker, and
   retry-with-backoff fire automatically. The factory's per-worker
   instantiation means each worker gets its own state (cap, circuit
   counter), so blowing the 10-K worker's cap doesn't affect the 8-K
   worker.
4. **Extended thinking auto-enable on Sonnet/Opus** — when the routed
   model is a reasoning-tier model (Sonnet or Opus), the call enables
   extended thinking with a fixed `EXTENDED_THINKING_BUDGET` token
   budget. Haiku-tier calls (templated extraction, pattern matching)
   skip thinking — no quality gain, pure latency cost.

What the wrapper deliberately doesn't do:

- **Batch API support** — `anthropic.messages.batches.*` has a
  different request/response shape (file-based + polling). It belongs
  in a follow-up; muddling sync + batch here would double the surface.
- **LangChain `ChatAnthropic` integration** — would force cost
  accounting through `AIMessage.usage_metadata` (different shape) and
  obscure `cache_control` behind `additional_kwargs`. The LangGraph
  research agent (Issue #28) calls this wrapper from its node bodies
  directly; node functions are just Python functions.
- **Cost logging plumbing** — OpenLLMetry / traceloop-sdk
  auto-instruments the Anthropic SDK and captures token counts on the
  active OTel span. We add only the missing piece: `llm.cost.est_usd`,
  computed from `usage_for_message` (the same pricing table cost_cap
  uses). No new logging system.

Production callers compose:

    _CLIENT = make_extraction_client(worker="s_filings", usd_cap=10.0)

    def extract_s_filing(raw_doc, prompt_version):
        response = _CLIENT(
            task="dilution_event",
            system_prompt=PROMPT.text,
            user_content=raw_doc.text,
            output_schema=SFilingOutput,
        )
        return validate_or_quarantine(parse(response), raw_doc.text, ...)
"""

from __future__ import annotations

from typing import Any, Final, Protocol

import anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from opentelemetry import trace
from pydantic import BaseModel

from auto_research._models import route_model
from auto_research._pricing import usd_for_message
from auto_research.agents.reliability import reliable_agent_node
from auto_research.extract._caching import cached_system_block

# The tool name every worker forces via `tool_choice`. Single value
# (not a per-worker name) so the response-parsing branch in
# `workers/_common.py` can match by literal without threading the
# worker name through.
RECORD_EXTRACTION_TOOL_NAME: Final[str] = "record_extraction"

# Token budget for extended thinking on Sonnet/Opus routes. 2048 is the
# Anthropic-documented sweet spot for structured-extraction tasks where
# the model benefits from reasoning before emitting JSON but doesn't
# need the deeper budgets reserved for math/code. Per the API contract,
# `max_tokens` must exceed this budget; the workers' default
# `max_tokens=4096` leaves 2048 tokens for actual output, which is
# more than enough for these schemas.
EXTENDED_THINKING_BUDGET: Final[int] = 2048


class ExtractionFn(Protocol):
    """Type of the callable returned by `make_extraction_client`.

    Documented as a Protocol so workers can annotate `_CLIENT: ExtractionFn`
    without having to spell out the kwargs each time.

    `output_schema` switches the response shape:

    - When provided (default for extraction workers), the schema is
      forwarded as the `record_extraction` tool's `input_schema` and
      `tool_choice` forces the model to emit exactly one tool_use
      block whose `.input` is the parsed dict. The worker reads it
      directly without a text/json.loads round-trip.
    - When `None` (default for callers that want free-form text — e.g.,
      the contextual-chunker writing a one-line context per child),
      the call omits `tools` / `tool_choice` and the response carries
      ordinary `text` content blocks.
    """

    def __call__(
        self,
        *,
        task: str,
        system_prompt: str,
        user_content: str,
        output_schema: type[BaseModel] | None = None,
        max_tokens: int = ...,
    ) -> Message: ...


def make_extraction_client(
    *,
    worker: str,
    usd_cap: float = 5.00,
    failures: int = 3,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    anthropic_client: anthropic.Anthropic | None = None,
) -> ExtractionFn:
    """Build a per-worker extraction callable with reliability + caching.

    Each call to this factory creates a fresh closure with its own
    reliability state — separate cost-cap totals, separate circuit-breaker
    counts. Production code instantiates one client per worker module
    (e.g., `_CLIENT = make_extraction_client(worker="s_filings", usd_cap=10.0)`
    at module top level) so the per-worker budgets are independent.

    `anthropic_client` is injected for hermetic testing — production
    callers omit it and get a real `anthropic.Anthropic()` with default
    `max_retries=2` (the SDK's internal retry layer, beneath our outer
    `@retry_with_backoff`).

    Args:
        worker: feeds `route_model(worker, task)`; also tags the
            reliability decorators' state for debugging.
        usd_cap: hard USD spend cap enforced by `@cost_cap` across all
            calls through this client.
        failures: consecutive-failure threshold for `@circuit_breaker`.
        max_retries: additional attempts after the first for
            `@retry_with_backoff` on 429 / 5xx / transient httpx errors.
        initial_wait, max_wait: exponential-jitter backoff bounds.
        anthropic_client: optional injected SDK client (for testing).
    """
    sdk = anthropic_client if anthropic_client is not None else anthropic.Anthropic()

    @reliable_agent_node(
        failures=failures,
        usd=usd_cap,
        max_retries=max_retries,
        initial_wait=initial_wait,
        max_wait=max_wait,
    )
    def _call(
        *,
        task: str,
        system_prompt: str,
        user_content: str,
        output_schema: type[BaseModel] | None = None,
        max_tokens: int = 4096,
    ) -> Message:
        # Route the model first — surfaces unknown-task ValueError before
        # we touch the network. `route_model` raises with a descriptive
        # message naming the bad (worker, task) pair.
        model = route_model(worker, task)

        # Extended thinking on Sonnet/Opus only. Haiku-tier templated
        # extraction is "high-volume pattern recognition" per §7.3 and
        # gains nothing from a thinking budget — pure latency cost. The
        # Anthropic-required precondition (max_tokens > budget_tokens)
        # holds by construction: workers' default max_tokens=4096
        # leaves 2048 for actual output, which fits every current
        # output schema.
        extra_kwargs: dict[str, Any] = {}
        if model.startswith(("claude-sonnet-", "claude-opus-")):
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": EXTENDED_THINKING_BUDGET,
            }

        # Server-side schema enforcement via tool_use when an
        # `output_schema` is provided. The tool's `input_schema` is the
        # pydantic model's JSON schema; Anthropic rejects responses
        # whose tool_use.input doesn't conform, cutting json-decode and
        # schema-noncompliance quarantines from ~5-15% to <1%. The
        # forced `tool_choice` makes the model emit exactly one
        # `record_extraction` tool_use block; the worker's
        # response-parser pulls `tool_use.input` directly without a
        # json.loads round-trip. When `output_schema` is None
        # (contextual-chunker and similar free-form text callers), the
        # call falls back to ordinary text content with no tool plumbing.
        if output_schema is not None:
            tool: ToolParam = {
                "name": RECORD_EXTRACTION_TOOL_NAME,
                "description": (
                    "Emit the structured extraction result. Call this tool "
                    "exactly once; its input is the full output object."
                ),
                "input_schema": output_schema.model_json_schema(),
            }
            tool_choice: ToolChoiceToolParam = {
                "type": "tool",
                "name": RECORD_EXTRACTION_TOOL_NAME,
            }
            extra_kwargs["tools"] = [tool]
            extra_kwargs["tool_choice"] = tool_choice

        # `cached_system_block` builds the structured-block form with
        # `cache_control: ephemeral` — same helper as the batch client
        # uses, so the W1 caching policy (system always cacheable, user
        # content uncached) stays consistent across both regimes.
        response = sdk.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=cached_system_block(system_prompt),  # type: ignore[arg-type]
            messages=[{"role": "user", "content": user_content}],
            **extra_kwargs,
        )

        # Emit per-call USD + token-count attributes on the active OTel
        # span. OpenLLMetry auto-instruments token counts when its
        # exporter is wired, but production OTel pipelines downstream
        # (and our tests) don't always run auto-instrumentation — we
        # set the attributes explicitly so the prompt-cache effect is
        # visible in dashboards regardless. `get_current_span()`
        # returns a no-op span when no tracer provider is configured,
        # so this is safe in tests that don't set one up.
        #
        # The cache_* counters are how we verify the system-prompt
        # cache marker is actually firing — `cache_read_input_tokens`
        # > 0 on the second-and-later calls within a 5-minute window
        # means the prefix hit the cache; a regression that strips
        # `cache_control: ephemeral` would silently zero these out
        # and triple our per-call cost without any other signal.
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

    return _call


__all__ = [
    "RECORD_EXTRACTION_TOOL_NAME",
    "ExtractionFn",
    "make_extraction_client",
]
