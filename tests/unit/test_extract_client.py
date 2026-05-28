"""Unit tests for `auto_research.extract.client` (Issue #10).

What's pinned here:

- `make_extraction_client` returns a per-worker callable with the
  contract-mandated reliability decorators composed in.
- The callable routes the model via `route_model(worker, task)` — no
  hardcoded `claude-...` literals reach the wrapped Anthropic SDK call.
- The system prompt is marked `cache_control: {"type": "ephemeral"}` by
  default (the opinionated W1 choice; ~80% of input cached per spec §7.4).
- Per-call USD cost is emitted as an OpenTelemetry span attribute
  (`llm.cost.est_usd`). OpenLLMetry auto-instrumentation captures the
  token counts; the USD pricing is on us because the SDK doesn't carry it.
- Reliability composition works: cost_cap actually trips, circuit_breaker
  actually opens, unknown-task lookup actually raises.

The Anthropic client is injected as a parameter for hermetic testing —
production callers omit it and get a real `anthropic.Anthropic()`.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from pydantic import BaseModel

from auto_research.agents.reliability import CircuitOpen, CostCapExceeded
from auto_research.extract.client import (
    RECORD_EXTRACTION_TOOL_NAME,
    make_extraction_client,
)


class _TinyOutput(BaseModel):
    """Minimal pydantic model used as `output_schema=` in client tests.

    Production callers pass their real Pydantic output schema; the
    client only uses it to build `tools[input_schema]`, so any model
    works for the request-shape assertions in this file.
    """

    answer: str

# --- fakes -----------------------------------------------------------------


def _make_message(
    *,
    model: str = "claude-haiku-4-5",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text="ok", citations=None)],
        model=model,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation=None,
            cache_creation_input_tokens=cache_creation or None,
            cache_read_input_tokens=cache_read or None,
            inference_geo=None,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


class _FakeAnthropicMessages:
    """Minimal stand-in for `anthropic.Anthropic().messages` that captures
    every call's kwargs so tests can assert on them.
    """

    def __init__(self, response_factory: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response_factory = response_factory or (lambda **kw: _make_message(model=kw["model"]))

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return self._response_factory(**kwargs)


class _FakeAnthropicClient:
    def __init__(self, response_factory: Any = None) -> None:
        self.messages = _FakeAnthropicMessages(response_factory)


def _as_sdk(fake: _FakeAnthropicClient) -> anthropic.Anthropic:
    """Cast a structural-fake into the SDK type for dependency injection.

    Production callers omit `anthropic_client`. Tests inject a duck-typed
    fake that's compatible with the slice of the SDK the wrapper uses
    (`.messages.create(...)`). `cast()` documents the test-double pattern
    without needing `# type: ignore` at every call site.
    """
    return cast(anthropic.Anthropic, fake)


# --- factory + routing -----------------------------------------------------


def test_factory_returns_callable() -> None:
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=10.00,
        anthropic_client=_as_sdk(_FakeAnthropicClient()),
    )
    assert callable(client)


def test_call_routes_model_via_route_model() -> None:
    fake = _FakeAnthropicClient()
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=10.00,
        anthropic_client=_as_sdk(fake),
    )
    client(task="dilution_event", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    # `s_filings.dilution_language` ⇒ Haiku 4.5 per spec §7.3.
    assert fake.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_call_routes_sonnet_for_cross_doc_task() -> None:
    fake = _FakeAnthropicClient(
        response_factory=lambda **kw: _make_message(model=kw["model"]),
    )
    client = make_extraction_client(
        worker="ten_k",
        usd_cap=10.00,
        anthropic_client=_as_sdk(fake),
    )
    client(task="supplier_mentions", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    assert fake.messages.calls[0]["model"] == "claude-sonnet-4-6"


def test_extended_thinking_auto_enabled_on_sonnet() -> None:
    """Sonnet-tier extraction (cross-doc reasoning per §7.3) gets extended
    thinking automatically — the worker doesn't have to remember it.
    Haiku-tier templated extraction does NOT get thinking (no quality
    benefit, pure latency cost). Also pins that `tools` + `tool_choice`
    are present alongside `thinking` so a future refactor that drops
    one when the other is set surfaces here instead of silently
    breaking Sonnet extraction.
    """
    fake = _FakeAnthropicClient(
        response_factory=lambda **kw: _make_message(model=kw["model"]),
    )
    client = make_extraction_client(
        worker="ten_k",
        usd_cap=10.00,
        anthropic_client=_as_sdk(fake),
    )
    # Sonnet route → thinking enabled
    client(task="supplier_mentions", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    sonnet_call = fake.messages.calls[0]
    assert sonnet_call["model"] == "claude-sonnet-4-6"
    assert "thinking" in sonnet_call
    assert sonnet_call["thinking"]["type"] == "enabled"
    assert sonnet_call["thinking"]["budget_tokens"] == 2048
    # thinking + tool_use compose together — both must be present on
    # the Sonnet path.
    assert "tools" in sonnet_call
    assert sonnet_call["tools"][0]["name"] == RECORD_EXTRACTION_TOOL_NAME
    assert sonnet_call["tool_choice"] == {
        "type": "tool",
        "name": RECORD_EXTRACTION_TOOL_NAME,
    }


def test_extended_thinking_skipped_on_haiku() -> None:
    fake = _FakeAnthropicClient(
        response_factory=lambda **kw: _make_message(model=kw["model"]),
    )
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=10.00,
        anthropic_client=_as_sdk(fake),
    )
    client(task="dilution_event", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    haiku_call = fake.messages.calls[0]
    assert haiku_call["model"] == "claude-haiku-4-5"
    assert "thinking" not in haiku_call


def test_call_sends_tool_use_payload_from_output_schema() -> None:
    """The client forwards `output_schema` as the tool's `input_schema`
    and forces `tool_choice={'type':'tool','name':'record_extraction'}`.
    Without this the worker's response-parser (which extracts
    `tool_use.input` directly) has nothing to read.
    """
    fake = _FakeAnthropicClient()
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=10.00,
        anthropic_client=_as_sdk(fake),
    )
    client(
        task="dilution_event",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    call = fake.messages.calls[0]
    # Pin the load-bearing parts: the tool name (matched by worker
    # parsers) and the input_schema (server-side validation contract).
    # Description is a human-facing string — a `description in tool`
    # check keeps it intentional without making prompt-engineering
    # edits churn this test.
    assert len(call["tools"]) == 1
    tool = call["tools"][0]
    assert tool["name"] == RECORD_EXTRACTION_TOOL_NAME
    assert tool["input_schema"] == _TinyOutput.model_json_schema()
    assert "description" in tool and isinstance(tool["description"], str) and tool["description"]
    assert call["tool_choice"] == {
        "type": "tool",
        "name": RECORD_EXTRACTION_TOOL_NAME,
    }


def test_call_raises_on_unknown_task() -> None:
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=10.00,
        anthropic_client=_as_sdk(_FakeAnthropicClient()),
    )
    with pytest.raises(ValueError) as exc_info:
        client(task="not_a_task", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    # The error from `route_model` names both the worker and the bad task
    # so a typo at the call site is obvious from the traceback.
    assert "not_a_task" in str(exc_info.value)
    assert "s_filings" in str(exc_info.value)


# --- caching policy --------------------------------------------------------


def test_system_prompt_marked_ephemeral_by_default() -> None:
    """The W1 opinionated choice: system prompt is *always* cacheable.

    Spec §7.4 economics ("~80% of input cached at \\$0.30/M") only work
    if the long stable prefix is the system prompt. Workers that need
    different caching breakpoints should drop down to the raw SDK,
    not pass kwargs that fight the wrapper.
    """
    fake = _FakeAnthropicClient()
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=10.00,
        anthropic_client=_as_sdk(fake),
    )
    client(task="dilution_event", system_prompt="long stable prompt", user_content="doc", output_schema=_TinyOutput)
    system = fake.messages.calls[0]["system"]
    # Shape: list[{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]
    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0]["type"] == "text"
    assert system[0]["text"] == "long stable prompt"
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_user_content_not_marked_cacheable() -> None:
    fake = _FakeAnthropicClient()
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=10.00,
        anthropic_client=_as_sdk(fake),
    )
    client(task="dilution_event", system_prompt="sys", user_content="doc text", output_schema=_TinyOutput)
    messages = fake.messages.calls[0]["messages"]
    # User message is plain — caching it would be wasteful because each
    # doc differs.
    assert messages == [{"role": "user", "content": "doc text"}]


# --- cost emission via OpenTelemetry --------------------------------------


def test_emits_est_usd_to_active_span() -> None:
    """`llm.cost.est_usd` is added to the active span on every call.

    OpenLLMetry's auto-instrumentation already captures token counts; the
    USD figure is the missing piece because pricing isn't in the SDK
    response. We add it to the same span so Langfuse + Grafana can read
    it without a join.
    """
    fake = _FakeAnthropicClient(
        response_factory=lambda **kw: _make_message(
            model=kw["model"],
            input_tokens=1_000_000,
            output_tokens=500_000,
        ),
    )
    mock_span = MagicMock()
    with patch(
        "auto_research.extract.client.trace.get_current_span",
        return_value=mock_span,
    ):
        client = make_extraction_client(
            worker="s_filings",
            usd_cap=1000.00,  # don't care about cap here
            anthropic_client=_as_sdk(fake),
        )
        client(task="dilution_event", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)

    # Haiku 4.5: \$1/MTok input + \$5/MTok output = 1 * 1.0 + 0.5 * 5.0 = \$3.50
    mock_span.set_attribute.assert_any_call("llm.cost.est_usd", pytest.approx(3.50))


def test_emits_token_and_cache_attributes_to_active_span() -> None:
    """Token counts + cache_creation/cache_read attributes are set on the
    active span so dashboards can verify the system-prompt cache marker
    is actually firing. Without `cache_read_input_tokens` visibility a
    regression that strips `cache_control: ephemeral` would silently
    triple per-call cost.
    """
    fake = _FakeAnthropicClient(
        response_factory=lambda **kw: _make_message(
            model=kw["model"],
            input_tokens=500,
            output_tokens=200,
            cache_creation=0,
            cache_read=1500,  # warm cache hit on the system prefix
        ),
    )
    mock_span = MagicMock()
    with patch(
        "auto_research.extract.client.trace.get_current_span",
        return_value=mock_span,
    ):
        client = make_extraction_client(
            worker="s_filings",
            usd_cap=1000.00,
            anthropic_client=_as_sdk(fake),
        )
        client(task="dilution_event", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)

    mock_span.set_attribute.assert_any_call("llm.input_tokens", 500)
    mock_span.set_attribute.assert_any_call("llm.output_tokens", 200)
    mock_span.set_attribute.assert_any_call("llm.cache_read_input_tokens", 1500)


def test_omits_cache_attributes_when_sdk_reports_none() -> None:
    """A response with no cache_read / cache_creation populated (cold
    call, first within a window) must not set the attributes to a
    spurious 0 — the absence is itself information that there was no
    cache activity."""
    fake = _FakeAnthropicClient(
        response_factory=lambda **kw: _make_message(
            model=kw["model"], cache_creation=0, cache_read=0
        ),
    )
    mock_span = MagicMock()
    with patch(
        "auto_research.extract.client.trace.get_current_span",
        return_value=mock_span,
    ):
        client = make_extraction_client(
            worker="s_filings",
            usd_cap=1000.00,
            anthropic_client=_as_sdk(fake),
        )
        client(task="dilution_event", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)

    keys_set = [c.args[0] for c in mock_span.set_attribute.call_args_list]
    assert "llm.cache_read_input_tokens" not in keys_set
    assert "llm.cache_creation_input_tokens" not in keys_set


# --- reliability decorators (composed via @reliable_agent_node) ----------


def test_cost_cap_trips_after_threshold_exceeded() -> None:
    # One Sonnet 4.6 call costing 1M+1M tokens = \$3 + \$15 = \$18.
    # Cap at \$5 → first call passes, second trips.
    fake = _FakeAnthropicClient(
        response_factory=lambda **kw: _make_message(
            model=kw["model"],
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        ),
    )
    client = make_extraction_client(
        worker="ten_k",
        usd_cap=5.00,
        anthropic_client=_as_sdk(fake),
    )
    client(task="supplier_mentions", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    with pytest.raises(CostCapExceeded):
        client(task="supplier_mentions", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)


def test_circuit_breaker_opens_after_consecutive_failures() -> None:
    # Inner SDK call always raises; after N consecutive failures the
    # circuit opens and the next call short-circuits with CircuitOpen.
    def boom(**kw: Any) -> Message:
        raise RuntimeError("synthetic upstream failure")

    fake = _FakeAnthropicClient(response_factory=boom)
    client = make_extraction_client(
        worker="s_filings",
        usd_cap=1000.00,
        failures=2,
        max_retries=0,  # don't burn the failure budget on retries
        initial_wait=0.0,
        max_wait=0.0,
        anthropic_client=_as_sdk(fake),
    )
    for _ in range(2):
        with pytest.raises(RuntimeError):
            client(task="dilution_event", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    with pytest.raises(CircuitOpen):
        client(task="dilution_event", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)


# --- factory ergonomics ----------------------------------------------------


def test_factory_per_worker_state_is_isolated() -> None:
    """Two factories build two independent clients with separate
    reliability state — exceeding the cap on the ten_k client should
    not affect the s_filings client.
    """
    fake_ten_k = _FakeAnthropicClient(
        response_factory=lambda **kw: _make_message(
            model=kw["model"], input_tokens=1_000_000, output_tokens=1_000_000
        ),
    )
    fake_s_filings = _FakeAnthropicClient()
    ten_k = make_extraction_client(
        worker="ten_k", usd_cap=5.00, anthropic_client=_as_sdk(fake_ten_k)
    )
    s_filings = make_extraction_client(
        worker="s_filings", usd_cap=5.00, anthropic_client=_as_sdk(fake_s_filings)
    )
    # Blow ten_k's cap.
    ten_k(task="supplier_mentions", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    with pytest.raises(CostCapExceeded):
        ten_k(task="supplier_mentions", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    # s_filings is unaffected.
    s_filings(task="dilution_event", system_prompt="sys", user_content="doc", output_schema=_TinyOutput)
    assert len(fake_s_filings.messages.calls) == 1
