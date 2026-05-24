"""Anthropic pricing constants and per-message USD computation.

Promoted out of `auto_research.agents.reliability` so two callers can use it
cleanly:

- `reliability.cost_cap` accumulates USD across a session for hard-cap
  enforcement.
- `extract.client` emits per-call USD as an OpenTelemetry span attribute
  (the SDK's auto-instrumentation captures token counts; USD pricing is
  on us).

Adding `_pricing.py` to the small `_io.py` / `_transport.py` family of
top-level shared utilities — three callers' worth of "this isn't
ingest-specific, isn't extract-specific, isn't agent-specific" code now
live in the package root.

W2 follow-up: source `_PRICING_PER_MTOK` from the Langfuse model registry
at startup so list-price drift doesn't require code edits.
"""

from __future__ import annotations

from typing import Final

from anthropic.types import Message

# Anthropic list prices (USD per million tokens) for the three tiers the
# spec routes to. Cached input is billed at 10% of base input per Anthropic's
# documented prompt-caching schedule; cache *writes* are billed at 125% of
# base input. We carry the full schedule so cost accounting stays truthful
# when prompt caching is enabled (Issue #10+).
#
# Source: Anthropic public pricing, last verified 2026-05-24.
_PRICING_PER_MTOK: Final[dict[str, tuple[float, float]]] = {
    # model: (input USD / MTok, output USD / MTok)
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-7": (15.00, 75.00),
}

_CACHE_READ_DISCOUNT: Final[float] = 0.10  # cached input billed at 10% of base.
_CACHE_WRITE_PREMIUM: Final[float] = 1.25  # cache *writes* billed at 125% of base.
# Per spec §7.4 backfill economics, ~2,700 docs are extracted via the Batch
# API at 50% off list. `response.usage.service_tier == "batch"` is the
# authoritative signal; ignoring it would have `cost_cap` trip ~2x too early
# on every nightly batch run.
_BATCH_DISCOUNT: Final[float] = 0.50


def usd_for_message(message: Message) -> float:
    """Compute the USD cost of a single `Message` response from its `usage`.

    Raises `KeyError` if `message.model` isn't in `_PRICING_PER_MTOK` —
    silently treating an unknown model as \\$0 would let any cost-cap leak.

    `priority` tier is 2x list but the spec doesn't route to it; leaving it
    at list price means callers *over-bill* (trip caps early) rather than
    under-bill — safe direction. Revisit if W2 adopts priority routing.
    """
    input_per_mtok, output_per_mtok = _PRICING_PER_MTOK[message.model]
    usage = message.usage

    base_input = usage.input_tokens
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0

    cost = 0.0
    cost += (base_input / 1_000_000) * input_per_mtok
    cost += (cache_read / 1_000_000) * input_per_mtok * _CACHE_READ_DISCOUNT
    cost += (cache_write / 1_000_000) * input_per_mtok * _CACHE_WRITE_PREMIUM
    cost += (usage.output_tokens / 1_000_000) * output_per_mtok

    if usage.service_tier == "batch":
        cost *= _BATCH_DISCOUNT
    return cost


__all__ = ["usd_for_message"]
