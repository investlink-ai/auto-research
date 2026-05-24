"""Shared Anthropic message-block builders for extract clients.

The sync `client.py` and batch `batch_client.py` both need to construct
a `system` block with `cache_control: ephemeral`. Two callers — extract
once to a shared helper rather than have the dict shape drift between
modules. The W1 opinionated caching policy (system prompt always
cacheable; user content uncached) lives here so changes apply
consistently to both regimes.

Workers needing different caching breakpoints (e.g., the long-doc
chunked-extraction RAG flow in W2) should call the SDK directly, not
extend this helper — its purpose is to make the common case foolproof,
not to expose every cache_control knob.
"""

from __future__ import annotations

from typing import Any


def cached_system_block(system_prompt: str) -> list[dict[str, Any]]:
    """Build the `system` parameter for `messages.create` /
    `messages.batches.create` with ephemeral caching on.

    Returned as a single-element list so the SDK reads it as the
    structured block form (which is what enables `cache_control` —
    the plain string form has no place to attach the cache hint).
    """
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]


__all__ = ["cached_system_block"]
