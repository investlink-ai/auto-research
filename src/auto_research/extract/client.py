"""Anthropic-SDK wrapper for extraction workers (Issue #10).

A thin layer on top of `anthropic.Anthropic` that bakes in the three
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
        )
        return validate_or_quarantine(parse(response), raw_doc.text, ...)
"""

from __future__ import annotations

from typing import Protocol

import anthropic
from anthropic.types import Message
from opentelemetry import trace

from auto_research._models import route_model
from auto_research._pricing import usd_for_message
from auto_research.agents.reliability import reliable_agent_node


class ExtractionFn(Protocol):
    """Type of the callable returned by `make_extraction_client`.

    Documented as a Protocol so workers can annotate `_CLIENT: ExtractionFn`
    without having to spell out the kwargs each time.
    """

    def __call__(
        self,
        *,
        task: str,
        system_prompt: str,
        user_content: str,
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
        max_tokens: int = 4096,
    ) -> Message:
        # Route the model first — surfaces unknown-task ValueError before
        # we touch the network. `route_model` raises with a descriptive
        # message naming the bad (worker, task) pair.
        model = route_model(worker, task)

        # Mark the system prompt as ephemeral. This is the W1 opinionated
        # default: the long stable prefix is always the system prompt;
        # per-doc user content is the variable part. Any worker needing
        # different caching breakpoints (rare — only the chunked-RAG flow
        # in W2) bypasses this wrapper and calls the SDK directly.
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        response = sdk.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": user_content}],
        )

        # Emit per-call USD on the active OTel span. OpenLLMetry's
        # auto-instrumentation has already populated input/output/cache
        # token counts; we add the USD figure because the SDK doesn't
        # carry pricing. `get_current_span()` returns a no-op span when
        # no tracer provider is configured, so this is safe in tests
        # that don't set one up.
        trace.get_current_span().set_attribute(
            "llm.cost.est_usd",
            usd_for_message(response),
        )

        return response

    return _call


__all__ = ["ExtractionFn", "make_extraction_client"]
