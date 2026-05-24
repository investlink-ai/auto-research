"""Thin Langfuse wrapper for the prompt registry (Issue #11).

Two operations:

- `register_prompt(name, version, text)` — push a code-defined prompt
  into Langfuse so the registry has version history. Idempotent on the
  Langfuse side (it dedupes on prompt text + label set).
- `set_prompt_tag(name, version, tag)` — flip a Langfuse label
  (`dev` / `staging` / `production`) onto a specific version. Used by
  `scripts/promote_prompt.py` after a gold-set eval gate passes.

What this module deliberately doesn't do:

- Fetch prompts at runtime. Workers read the in-code `*_PROMPT` /
  `*_PROMPT_VERSION` constants. A network round-trip per extraction
  would make the cache key non-deterministic (Langfuse latency) and
  introduce a runtime dep on a service that's allowed to be down.
- Diff in-code prompt vs Langfuse-registered prompt. That's the job of
  CI or a manual `python -m ...prompts._registry_list` invocation; this
  module stays small.

The Langfuse v2 SDK is sync; W1 doesn't need async for one-shot
registration calls.
"""

from __future__ import annotations

from typing import Any, Protocol


class _LangfuseClient(Protocol):
    """Minimal slice of `langfuse.Langfuse` we depend on. Lets a
    `MagicMock()` stand in for tests without a runtime Langfuse instance."""

    def create_prompt(self, **kwargs: Any) -> Any: ...
    def update_prompt(self, **kwargs: Any) -> Any: ...


def register_prompt(
    *,
    name: str,
    version: str,
    text: str,
    client: _LangfuseClient,
) -> None:
    """Push a code-defined prompt into the Langfuse registry."""
    client.create_prompt(
        name=name,
        prompt=text,
        labels=[version],
        type="text",
    )


def set_prompt_tag(
    *,
    name: str,
    version: str,
    tag: str,
    client: _LangfuseClient,
) -> None:
    """Flip a tag label (`dev`/`staging`/`production`) onto a version."""
    client.update_prompt(
        name=name,
        version=version,
        new_labels=[tag],
    )


__all__ = ["register_prompt", "set_prompt_tag"]
