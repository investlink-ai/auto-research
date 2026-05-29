"""Anthropic-SDK wrapper for extraction workers.

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
  research agent calls this wrapper from its node bodies directly;
  node functions are just Python functions.
- **A normalized cross-provider response dataclass** — see
  `learning/2026-05-28-extraction-pipeline-cost-model.md` §10 for the
  full discussion. The `(text | dict | None, UsageDict)` tuple shape
  is the smallest abstraction that lets the same Protocol back both
  Anthropic and OpenAI-compat (Ollama / vLLM) without leaking
  provider-specific types across the wrapper boundary.

The reliability decorators wrap a Message-returning *inner* call so
`cost_cap` continues to read `Message.usage` for per-call USD
accounting. The public `ExtractionFn` callable wraps that inner with
a tuple-conversion outer that lifts `tool_use.input` / text-block
content + `usage.*` into the provider-agnostic shape.

Production callers compose:

    _CLIENT = make_extraction_client(worker="s_filings", usd_cap=10.0)

    def extract_s_filing(raw_doc, prompt_version):
        parsed, usage = _CLIENT(
            task="dilution_event",
            system_prompt=PROMPT.text,
            user_content=raw_doc.text,
            output_schema=SFilingOutput,
        )
        if parsed is None:
            return None  # no tool_use block — quarantine
        return validate_or_quarantine(
            SFilingOutput.model_validate(parsed),
            raw_doc.text, ...,
        )
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
from auto_research.extract._response import UsageDict

# The single structured-output tool name offered to every worker. On
# non-thinking (Haiku) routes the wrapper forces it via `tool_choice`; on
# thinking-enabled (Sonnet/Opus) routes it is offered with the default
# `auto` choice (forcing is incompatible with extended thinking). One fixed
# value (not a per-worker name) lets the response-parsing branch match by
# literal without threading the worker name through.
RECORD_EXTRACTION_TOOL_NAME: Final[str] = "record_extraction"


def _extract_tool_use_input(message: Message) -> dict[str, Any] | None:
    """Return the first `record_extraction` tool_use block's `.input`.

    Returns `None` when the response carries no matching tool_use block
    (e.g., model refused, or the call ran in free-form text mode).
    The wrapper surfaces `None` to the caller through the tuple's
    first element so workers can quarantine without parsing Message
    themselves.

    The SDK's `ToolUseBlock.input` is typed and validated as a `dict`
    at parse-time — non-dict inputs are rejected by the SDK before
    they reach this helper, so the return type is a real `dict` (not
    `dict | object`).
    """
    for block in message.content:
        if block.type == "tool_use" and block.name == RECORD_EXTRACTION_TOOL_NAME:
            return block.input
    return None


def _join_text_blocks(message: Message) -> str:
    """Concatenate every text block in `message.content` with single spaces.

    Free-form text callers (today: the contextual chunker, which omits
    `output_schema=`) receive the joined string as the tuple's first
    element. Whitespace normalization (collapsing internal runs) is
    caller-specific and stays in the caller; this helper preserves the
    raw text faithfully.
    """
    return " ".join(b.text for b in message.content if b.type == "text")


def _message_to_usage_dict(message: Message) -> UsageDict:
    """Lift `Message.usage` + `Message.stop_reason` into the provider-agnostic
    `UsageDict`.

    `cache_*` fields are only set when the SDK reports a populated value —
    a `None` from the SDK means the cache wasn't touched on this call
    (cold first call, no `cache_control` block, or pre-cache model
    version), and a spurious `0` would lie to dashboards about cache
    activity. `stop_reason` is always carried for free-form callers
    that gate on it.
    """
    usage = message.usage
    out: UsageDict = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    if usage.cache_read_input_tokens is not None:
        out["cache_read_input_tokens"] = usage.cache_read_input_tokens
    if usage.cache_creation_input_tokens is not None:
        out["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
    if message.stop_reason is not None:
        out["stop_reason"] = message.stop_reason
    # Block-type fingerprint preserves the structural signal that an
    # operator triaging a quarantined no-payload record needs to tell
    # `["text"]` (model refused with prose) apart from
    # `["thinking", "text"]` (extended-thinking budget burned, no
    # tool_use). The list is short (1-3 entries per Anthropic
    # response shape), so we set it on every call rather than only on
    # the no-payload outcome — callers that don't quarantine simply
    # ignore it.
    out["response_block_types"] = [b.type for b in message.content]
    return out


# Token budget for extended thinking on Sonnet/Opus routes. 2048 is the
# Anthropic-documented sweet spot for structured-extraction tasks where
# the model benefits from reasoning before emitting JSON but doesn't
# need the deeper budgets reserved for math/code. Per the API contract,
# `max_tokens` must exceed this budget; the workers' default
# `max_tokens=4096` leaves 2048 tokens for actual output, which is
# more than enough for these schemas.
EXTENDED_THINKING_BUDGET: Final[int] = 2048


class ExtractionFn(Protocol):
    """Type of the callable returned by `make_extraction_client` (and by
    the OpenAI-compat factory). Workers annotate `_CLIENT: ExtractionFn`
    without spelling the kwargs out at every reference.

    Return shape: `tuple[dict[str, Any] | str | None, UsageDict]`.

    - When `output_schema` is provided (default for extraction
      workers), the first element is the structured output as a
      `dict` parsed from the provider-native structured-output channel
      (Anthropic `tool_use.input`; OpenAI `response_format=json_schema`
      content). `None` means the provider returned no usable
      structured payload — model refused, mid-emission truncation, or
      a free-form text response from a provider that doesn't honor
      `output_schema=`; callers route this to quarantine.

    - When `output_schema=None` (default for free-form text callers —
      e.g., the contextual chunker emitting a one-line context), the
      first element is the joined text content as a `str`.

    `UsageDict` carries token counts + cache markers + provider-raw
    `stop_reason`. See `_response.py` for the full shape.
    """

    def __call__(
        self,
        *,
        task: str,
        system_prompt: str,
        user_content: str,
        output_schema: type[BaseModel] | None = None,
        max_tokens: int = ...,
    ) -> tuple[dict[str, Any] | str | None, UsageDict]: ...


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
            calls through this client. The decorator reads
            `Message.usage` on the *inner* call (which still returns
            a Message) so per-call USD accounting works unchanged
            despite the public callable's tuple return.
        failures: consecutive-failure threshold for `@circuit_breaker`.
        max_retries: additional attempts after the first for
            `@retry_with_backoff` on 429 / 5xx / transient httpx errors.
        initial_wait, max_wait: exponential-jitter backoff bounds.
        anthropic_client: optional injected SDK client (for testing).
    """
    sdk = anthropic_client if anthropic_client is not None else anthropic.Anthropic()

    # Per-client memo of `output_schema.model_json_schema()`. Pydantic v2
    # does NOT cache that call (each invocation builds a fresh dict, ~57μs
    # per call locally measured), and the same 4-7 schemas are reused
    # across thousands of extraction calls during a backfill. Cache by
    # class identity so a re-defined schema (same name, new fields)
    # doesn't return a stale entry.
    _schema_cache: dict[int, dict[str, Any]] = {}

    def _tool_input_schema(output_schema: type[BaseModel]) -> dict[str, Any]:
        cached = _schema_cache.get(id(output_schema))
        if cached is None:
            cached = output_schema.model_json_schema()
            _schema_cache[id(output_schema)] = cached
        return cached

    @reliable_agent_node(
        failures=failures,
        usd=usd_cap,
        max_retries=max_retries,
        initial_wait=initial_wait,
        max_wait=max_wait,
    )
    def _call_inner(
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
        thinking_enabled = model.startswith(("claude-sonnet-", "claude-opus-"))
        if thinking_enabled:
            extra_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": EXTENDED_THINKING_BUDGET,
            }

        # Server-side schema enforcement via tool_use when an
        # `output_schema` is provided. The tool's `input_schema` is the
        # pydantic model's JSON schema; Anthropic rejects responses
        # whose tool_use.input doesn't conform, cutting json-decode and
        # schema-noncompliance quarantines from ~5-15% to <1%. The outer
        # wrapper pulls `tool_use.input` directly without a json.loads
        # round-trip. When `output_schema` is None (contextual-chunker
        # and similar free-form text callers), the call falls back to
        # ordinary text content with no tool plumbing.
        #
        # tool_choice is forced (`type: tool`) ONLY on non-thinking
        # (Haiku) routes — that guarantees exactly one `record_extraction`
        # block. The Anthropic API rejects a forced tool_choice while
        # extended thinking is enabled ("Thinking may not be enabled when
        # tool_choice forces tool use"), so on Sonnet/Opus routes the tool
        # is offered but tool_choice is left at its default (`auto`). The
        # single offered tool plus its description reliably elicits the
        # call; a miss surfaces as a `None` payload and routes to
        # quarantine, exactly like a refusal.
        if output_schema is not None:
            tool: ToolParam = {
                "name": RECORD_EXTRACTION_TOOL_NAME,
                "description": (
                    "Emit the structured extraction result. Call this tool "
                    "exactly once; its input is the full output object."
                ),
                "input_schema": _tool_input_schema(output_schema),
            }
            extra_kwargs["tools"] = [tool]
            if not thinking_enabled:
                tool_choice: ToolChoiceToolParam = {
                    "type": "tool",
                    "name": RECORD_EXTRACTION_TOOL_NAME,
                }
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

    def _call(
        *,
        task: str,
        system_prompt: str,
        user_content: str,
        output_schema: type[BaseModel] | None = None,
        max_tokens: int = 4096,
    ) -> tuple[dict[str, Any] | str | None, UsageDict]:
        # The reliability stack wraps `_call_inner` (Message return) so
        # `cost_cap` keeps reading `Message.usage` for per-call USD
        # accounting; the public callable handles tuple conversion
        # here. CostCapExceeded / CircuitOpen / route-model ValueError
        # propagate transparently through this layer.
        message = _call_inner(
            task=task,
            system_prompt=system_prompt,
            user_content=user_content,
            output_schema=output_schema,
            max_tokens=max_tokens,
        )
        usage = _message_to_usage_dict(message)
        if output_schema is not None:
            return _extract_tool_use_input(message), usage
        return _join_text_blocks(message), usage

    return _call


__all__ = [
    "RECORD_EXTRACTION_TOOL_NAME",
    "ExtractionFn",
    "make_extraction_client",
]
