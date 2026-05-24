"""Thin Langfuse wrapper for the prompt registry (Issue #11).

Two operations:

- `register_prompt(name, version, text)` — push a code-defined prompt
  into Langfuse so the registry has version history. Returns the Langfuse
  side's integer version, which we need because Langfuse's `update_prompt`
  takes `version: int` (the auto-assigned monotonic per-name index),
  while our code-side version is a string label like `"v1"`. Both
  dimensions exist; the code-side string lives in Langfuse only as a
  `labels` entry.

- `set_prompt_tag(name, version, tag)` — flip a label
  (`dev` / `staging` / `production`) onto the prompt whose code-side
  version-label matches `version`. The function looks up the matching
  Langfuse int version by label first, then calls `update_prompt`. A
  naive `update_prompt(version=str)` raises ValidationError because the
  SDK is typed `version: int`; this lookup is what makes the promotion
  flow work end-to-end.

What this module deliberately doesn't do:

- Fetch prompts at runtime for *extraction* workers. Workers read the
  in-code `*_PROMPT` / `*_PROMPT_VERSION` constants. A network round-trip
  per extraction would make the cache key non-deterministic (Langfuse
  latency) and introduce a runtime dep on a service that's allowed to
  be down.
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
    def get_prompt(self, name: str, **kwargs: Any) -> Any: ...
    def update_prompt(
        self, *, name: str, version: int, new_labels: list[str]
    ) -> Any: ...


def register_prompt(
    *,
    name: str,
    version: str,
    text: str,
    client: _LangfuseClient,
) -> int:
    """Push a code-defined prompt into the Langfuse registry.

    Returns the Langfuse-assigned **integer** version of the created
    prompt. The code-side `version` (a string like `"v1"`) is stored as a
    label on the Langfuse prompt, providing the lookup mechanism that
    `set_prompt_tag` relies on.
    """
    prompt_client = client.create_prompt(
        name=name,
        prompt=text,
        labels=[version],
        type="text",
    )
    return int(prompt_client.version)


def set_prompt_tag(
    *,
    name: str,
    version: str,
    tag: str,
    client: _LangfuseClient,
) -> None:
    """Flip a tag label (`dev`/`staging`/`production`) onto the Langfuse
    prompt whose code-side version-label matches `version`.

    Looks up the Langfuse int version by label first; calling
    `update_prompt(version=str)` directly raises ValidationError because
    the SDK is typed `version: int`.
    """
    prompt = client.get_prompt(name, label=version)
    int_version = int(prompt.version)
    client.update_prompt(
        name=name,
        version=int_version,
        new_labels=[tag],
    )


__all__ = ["register_prompt", "set_prompt_tag"]
