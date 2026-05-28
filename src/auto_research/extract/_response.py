"""Provider-agnostic response primitives for the extraction call surface.

The `ExtractionFn` Protocol (defined in `client.py`) returns
`tuple[dict | str | None, UsageDict]` so the same callable shape can
back two providers:

- **Anthropic SDK** (production today, `make_extraction_client`) —
  `tool_use.input` parsed to a `dict` when the caller passes
  `output_schema=`; joined text blocks as a `str` otherwise. Token /
  cache counters and `stop_reason` lifted out of `Message.usage` into
  `UsageDict`.
- **OpenAI-compatible HTTP** (Ollama / vLLM / MLX-server,
  `make_openai_compat_extraction_client`) — `response_format=json_schema`
  parsed to a `dict` when the caller passes `output_schema=`;
  `choices[0].message.content` as a `str` otherwise. `usage.*_tokens`
  and `finish_reason` lifted into `UsageDict`.

Why a TypedDict and not a normalized dataclass: the broader
discussion in
`learning/2026-05-28-extraction-pipeline-cost-model.md` §10 covers the
"why not a `ExtractionResponse` + `UnifiedUsage` dataclass" decision.
At our 1-2 provider regime, the dataclass abstraction is tax with no
unlock; `UsageDict` is the smallest shape that lets a wrapper surface
the token/stop_reason signal callers actually use.

The cache fields are `NotRequired` because OpenAI-compat local servers
don't report Anthropic-style cache metrics (vLLM has prefix caching
but exposes it differently; not normalized here). `stop_reason` is the
*provider-raw* value (Anthropic emits `end_turn`/`max_tokens`/`refusal`/
`pause_turn`; OpenAI emits `stop`/`length`/`tool_calls`/`content_filter`).
Consumers that care about it (today: the contextual chunker) match
against the value space they expect.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class UsageDict(TypedDict):
    """Provider-agnostic token-accounting payload returned alongside text/dict.

    `input_tokens` and `output_tokens` are required because they
    underpin per-call cost computation in both provider wrappers.

    `cache_read_input_tokens` / `cache_creation_input_tokens` are
    Anthropic-specific signals — they are how the OTel attribute
    `llm.cache_read_input_tokens > 0` canary in
    `learning/2026-05-28-extraction-pipeline-cost-model.md` §3.4
    proves the system-prompt cache is firing. Absent on OpenAI-compat.

    `stop_reason` is the provider-raw stop / finish reason. Anthropic:
    `{end_turn, stop_sequence, max_tokens, refusal, pause_turn, tool_use}`.
    OpenAI: `{stop, length, tool_calls, content_filter}`. Free-form
    text callers (the contextual chunker) drop the response when this
    value is outside their allow-set; structured callers ignore it.

    `response_block_types` carries the Anthropic content-block-type
    fingerprint (e.g. `["text"]` for a refusal-with-text response;
    `["thinking", "text"]` when extended thinking emitted a text-only
    completion). Populated by the Anthropic wrapper because it is the
    diagnostic that distinguishes "model refused" from "model
    exhausted its thinking budget" in a quarantine record — both
    surface to the worker as `parsed is None` but want different
    triage. Absent on OpenAI-compat (no per-block structure; the
    equivalent signal is `stop_reason="content_filter"` already on
    this dict).
    """

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: NotRequired[int]
    cache_creation_input_tokens: NotRequired[int]
    stop_reason: NotRequired[str]
    response_block_types: NotRequired[list[str]]


__all__ = ["UsageDict"]
