"""Unit tests for `auto_research.extract.openai_compat_client`.

What's pinned here:

- `make_openai_compat_extraction_client` returns a per-worker callable
  with the `ExtractionFn` Protocol shape (tuple of structured payload +
  `UsageDict`).
- The callable routes the model via `route_model(worker, task)` and
  strips the `local/` prefix before forwarding to the OpenAI-compat
  HTTP API.
- `output_schema=` forwards `response_format=json_schema` with the
  Pydantic JSON schema; the `record_extraction` name is shared with
  the Anthropic path's `tool_choice` (provider symmetry).
- Free-form calls (`output_schema=None`) skip `response_format` and
  surface the raw `choices[0].message.content` string.
- `UsageDict` lifts `prompt_tokens` / `completion_tokens` /
  `finish_reason` from the OpenAI response shape; `cache_*` fields are
  deliberately absent.
- Reliability composition works: circuit breaker actually opens on
  consecutive failures; retry handles transient errors (rate limit,
  5xx, connection error) but propagates the original 4xx programmer
  errors and `route_model` ValueError.

The OpenAI SDK is injected as a duck-typed fake — production callers
omit it and get a real `openai.OpenAI(base_url=...)`.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from auto_research.agents.reliability import CircuitOpen
from auto_research.extract.client import RECORD_EXTRACTION_TOOL_NAME
from auto_research.extract.openai_compat_client import (
    OLLAMA_DEFAULT_BASE_URL,
    make_openai_compat_extraction_client,
)

# --- fakes -----------------------------------------------------------------


class _TinyOutput(BaseModel):
    """Minimal pydantic model used as `output_schema=` in client tests."""

    answer: str


def _make_completion(
    *,
    model: str = "qwen3.5:9b",
    content: str | None = '{"answer": "ok"}',
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    include_usage: bool = True,
) -> ChatCompletion:
    """Build a `ChatCompletion` shaped like what an OpenAI-compat server
    emits. `include_usage=False` simulates a degenerate server that
    omits the usage block — the wrapper treats this as 0/0.
    """
    usage = (
        CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        if include_usage
        else None
    )
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model=model,
        choices=[
            Choice(
                finish_reason=finish_reason,  # type: ignore[arg-type]
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                ),
                logprobs=None,
            )
        ],
        usage=usage,
    )


class _FakeChatCompletions:
    """Minimal stand-in for `openai.OpenAI().chat.completions` that
    captures every call's kwargs so tests can assert on them.
    """

    def __init__(self, response_factory: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response_factory = response_factory or (
            lambda **kw: _make_completion(model=kw["model"])
        )

    def create(self, **kwargs: Any) -> ChatCompletion:
        self.calls.append(kwargs)
        return self._response_factory(**kwargs)


class _FakeChat:
    def __init__(self, response_factory: Any = None) -> None:
        self.completions = _FakeChatCompletions(response_factory)


class _FakeOpenAIClient:
    def __init__(self, response_factory: Any = None) -> None:
        self.chat = _FakeChat(response_factory)


def _as_sdk(fake: _FakeOpenAIClient) -> openai.OpenAI:
    """Cast a structural-fake into the SDK type for dependency injection."""
    return cast(openai.OpenAI, fake)


def _stub_route(worker: str, task: str) -> str:
    """Test-local routing-table stub that maps any (worker, task) to a
    `local/` model id. The dispatch test in
    `test_extract_local_dispatch.py` covers the production routing
    table's role; here we only want to exercise the wrapper.
    """
    return "local/qwen3.5:9b"


# --- factory + routing -----------------------------------------------------


def test_factory_returns_callable() -> None:
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(_FakeOpenAIClient()),
    )
    assert callable(client)


def test_factory_constructs_real_client_with_default_base_url() -> None:
    """When no `openai_client` is injected and `base_url=None`, the
    factory builds a real `openai.OpenAI` against the Ollama default
    endpoint. Pinning this catches a regression where the default
    silently switched (e.g., to OpenAI's hosted endpoint, leaking
    spend that should never have left the laptop)."""
    with patch("auto_research.extract.openai_compat_client.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        make_openai_compat_extraction_client(worker="contextual_chunking")
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == OLLAMA_DEFAULT_BASE_URL


def test_factory_constructs_real_client_with_custom_base_url() -> None:
    """A `base_url` kwarg flows through to the SDK constructor — the
    knob that lets a deploy point at vLLM (`:8000`) or MLX-server
    (custom port) without code changes."""
    with patch("auto_research.extract.openai_compat_client.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        make_openai_compat_extraction_client(
            worker="contextual_chunking",
            base_url="http://my-vllm:8000/v1",
            api_key="vllm-token",
        )
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "http://my-vllm:8000/v1"
        assert kwargs["api_key"] == "vllm-token"


def test_call_routes_model_via_route_model_and_strips_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`route_model` returns `local/qwen3.5:9b`; the wrapper strips
    `local/` before sending to the API because the OpenAI-compat
    endpoint expects the bare server-native ID."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient()
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    assert fake.chat.completions.calls[0]["model"] == "qwen3.5:9b"


def test_call_raises_on_unknown_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """`route_model`'s ValueError propagates — same loud-failure
    behavior as the Anthropic path."""
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(_FakeOpenAIClient()),
        max_retries=0,
    )
    with pytest.raises(ValueError) as exc_info:
        client(
            task="not_a_task",
            system_prompt="sys",
            user_content="doc",
            output_schema=_TinyOutput,
        )
    assert "not_a_task" in str(exc_info.value)
    assert "contextual_chunking" in str(exc_info.value)
    # Avoid an unused warning on the monkeypatch fixture (kept in
    # the signature for symmetry with the routing tests).
    _ = monkeypatch


# --- structured-output path ------------------------------------------------


def test_structured_call_forwards_response_format_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `output_schema` is set, the wrapper builds
    `response_format={"type": "json_schema", "json_schema": {name,
    schema, strict}}` from the Pydantic JSON schema — the OpenAI
    structured-outputs equivalent of the Anthropic path's
    `tool_choice=record_extraction`."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient()
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    rf = fake.chat.completions.calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    inner = rf["json_schema"]
    assert inner["name"] == RECORD_EXTRACTION_TOOL_NAME
    assert inner["schema"] == _TinyOutput.model_json_schema()
    assert inner["strict"] is True


def test_structured_call_returns_dict_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured payload returns as `dict`; `UsageDict` carries
    Anthropic-renamed token counts + `stop_reason` lifted from
    `finish_reason`."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient(
        response_factory=lambda **kw: _make_completion(
            model=kw["model"],
            content='{"answer": "structured-result"}',
            finish_reason="stop",
            prompt_tokens=200,
            completion_tokens=10,
        )
    )
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    parsed, usage = client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    assert parsed == {"answer": "structured-result"}
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 10
    assert usage.get("stop_reason") == "stop"
    assert "cache_read_input_tokens" not in usage
    assert "cache_creation_input_tokens" not in usage


def test_structured_call_returns_none_when_content_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compat servers return `content=None` for
    refusal / content-filter responses; the wrapper surfaces `None`
    so callers route to quarantine (mirroring the Anthropic "no
    tool_use block" outcome)."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient(
        response_factory=lambda **kw: _make_completion(
            model=kw["model"], content=None, finish_reason="content_filter"
        )
    )
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    parsed, usage = client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    assert parsed is None
    assert usage.get("stop_reason") == "content_filter"


def test_structured_call_returns_none_on_undecodable_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`response_format=json_schema` with `strict=True` should make
    invalid JSON unreachable, but older Ollama/vLLM builds degrade to
    best-effort emission. Undecodable content surfaces as `None` —
    same quarantine path as a refusal."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient(
        response_factory=lambda **kw: _make_completion(
            model=kw["model"], content="not-json-at-all"
        )
    )
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    parsed, _usage = client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    assert parsed is None


def test_structured_call_returns_none_on_top_level_non_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response that parses to a bare scalar or list (`"123"`,
    `"[]"`) is off-schema for the structured contract; treat it as no
    structured payload rather than passing a weakly-typed value to
    `output_model.model_validate` upstream."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient(
        response_factory=lambda **kw: _make_completion(
            model=kw["model"], content="[1, 2, 3]"
        )
    )
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    parsed, _usage = client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    assert parsed is None


# --- free-form text path ---------------------------------------------------


def test_freeform_call_omits_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`output_schema=None` is the contextual-chunker path — no
    `response_format`, free-text content, joined text returned."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient(
        response_factory=lambda **kw: _make_completion(
            model=kw["model"], content="This chunk discusses TSMC capacity."
        )
    )
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    text, usage = client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
    )
    assert "response_format" not in fake.chat.completions.calls[0]
    assert text == "This chunk discusses TSMC capacity."
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50


def test_freeform_call_returns_empty_string_on_null_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusal / null content collapses to the empty string on the
    free-form path; callers' drop logic treats empty text as a drop
    condition."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient(
        response_factory=lambda **kw: _make_completion(
            model=kw["model"], content=None
        )
    )
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    text, _usage = client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
    )
    assert text == ""


# --- system + user message wiring -----------------------------------------


def test_call_sends_system_and_user_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Anthropic-compat wrapper applies `cache_control: ephemeral`
    to the system block; the OpenAI-compat path can't (the v1
    chat-completions shape has no per-message cache control), so the
    system prompt lands as a plain message. Pin the shape so a
    regression that quietly tries to add a cache-control object
    (which the server would reject) surfaces."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient()
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    client(
        task="contextual_chunk",
        system_prompt="system-prompt-text",
        user_content="user-content-text",
        output_schema=_TinyOutput,
    )
    msgs = fake.chat.completions.calls[0]["messages"]
    assert msgs == [
        {"role": "system", "content": "system-prompt-text"},
        {"role": "user", "content": "user-content-text"},
    ]


# --- usage edge cases ------------------------------------------------------


def test_usage_dict_falls_back_to_zero_when_server_omits_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some local servers emit a degenerate response with no usage
    block; `UsageDict` falls back to 0/0 rather than raising — the
    honest fallback for "we don't know."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient(
        response_factory=lambda **kw: _make_completion(
            model=kw["model"], include_usage=False
        )
    )
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        openai_client=_as_sdk(fake),
    )
    _parsed, usage = client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


# --- reliability primitives ------------------------------------------------


def test_circuit_breaker_opens_after_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-retryable inner failures (ValueError) burn circuit slots;
    after N consecutive failures the wrapper raises `CircuitOpen`."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )

    def boom(**kw: Any) -> ChatCompletion:
        raise RuntimeError("synthetic upstream failure")

    fake = _FakeOpenAIClient(response_factory=boom)
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        failures=2,
        max_retries=0,
        initial_wait=0.0,
        max_wait=0.0,
        openai_client=_as_sdk(fake),
    )
    for _ in range(2):
        with pytest.raises(RuntimeError):
            client(
                task="contextual_chunk",
                system_prompt="sys",
                user_content="doc",
                output_schema=_TinyOutput,
            )
    with pytest.raises(CircuitOpen):
        client(
            task="contextual_chunk",
            system_prompt="sys",
            user_content="doc",
            output_schema=_TinyOutput,
        )


def test_retry_recovers_on_transient_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`httpx.ConnectError` is in `TRANSIENT_NETWORK_ERRORS`; the
    retry layer should swallow it once and succeed on the next attempt.
    This is the local-server-restarted-mid-call recovery path."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )

    attempts = {"n": 0}

    def flaky(**kw: Any) -> ChatCompletion:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("local server starting")
        return _make_completion(model=kw["model"])

    fake = _FakeOpenAIClient(response_factory=flaky)
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        max_retries=2,
        initial_wait=0.0,
        max_wait=0.0,
        openai_client=_as_sdk(fake),
    )
    parsed, _usage = client(
        task="contextual_chunk",
        system_prompt="sys",
        user_content="doc",
        output_schema=_TinyOutput,
    )
    assert parsed == {"answer": "ok"}
    assert attempts["n"] == 2


def test_retry_does_not_swallow_4xx_programmer_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4xx (`BadRequestError`, `AuthenticationError`,
    `NotFoundError`) signals a programmer / config mistake (wrong
    model id, missing key, malformed payload) — must propagate so it
    fails loud instead of burning the retry budget."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )

    def reject(**kw: Any) -> ChatCompletion:
        # `BadRequestError` is the 400 subclass. Build with the
        # minimal kwargs the SDK exposes on these exception types.
        raise openai.BadRequestError(
            message="model not found: qwen3.5:9b",
            response=httpx.Response(
                400, request=httpx.Request("POST", "http://localhost/v1")
            ),
            body=None,
        )

    fake = _FakeOpenAIClient(response_factory=reject)
    client = make_openai_compat_extraction_client(
        worker="contextual_chunking",
        max_retries=5,
        initial_wait=0.0,
        max_wait=0.0,
        openai_client=_as_sdk(fake),
    )
    with pytest.raises(openai.BadRequestError):
        client(
            task="contextual_chunk",
            system_prompt="sys",
            user_content="doc",
            output_schema=_TinyOutput,
        )


# --- OTel attribute emission ----------------------------------------------


def test_emits_local_model_id_to_active_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`llm.local_model_id` is the dashboard split between API-tier
    and local-tier calls; pin its emission so a refactor doesn't
    silently lose the signal."""
    monkeypatch.setattr(
        "auto_research.extract.openai_compat_client.route_model", _stub_route
    )
    fake = _FakeOpenAIClient()
    mock_span = MagicMock()
    with patch(
        "auto_research.extract.openai_compat_client.trace.get_current_span",
        return_value=mock_span,
    ):
        client = make_openai_compat_extraction_client(
            worker="contextual_chunking",
            openai_client=_as_sdk(fake),
        )
        client(
            task="contextual_chunk",
            system_prompt="sys",
            user_content="doc",
            output_schema=_TinyOutput,
        )

    mock_span.set_attribute.assert_any_call("llm.backend", "openai_compat")
    mock_span.set_attribute.assert_any_call("llm.local_model_id", "qwen3.5:9b")
    mock_span.set_attribute.assert_any_call("llm.input_tokens", 100)
    mock_span.set_attribute.assert_any_call("llm.output_tokens", 50)
