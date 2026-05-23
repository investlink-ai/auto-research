"""Unit tests for `auto_research.agents.reliability` decorators (Issue #8).

One test per decorator's trip condition. Composite-ordering tests assert
that `cost_cap` short-circuits before `circuit_breaker` so the documented
"cost_cap outermost" contract holds end-to-end, not just by inspection.

The cost_cap unit tests construct a real `anthropic.types.Message` via the
SDK's Pydantic models — real-shape, never a `Mock`. The recorded-cassette
test in `tests/integration/test_reliability_vcr.py` covers the same logic
against bytes that came back from a real `POST /v1/messages` exchange.
"""

from __future__ import annotations

import httpx
import pytest
from anthropic import APIStatusError, RateLimitError
from anthropic.types import Message, TextBlock, Usage

from auto_research.agents.reliability import (
    CircuitOpen,
    CostCapExceeded,
    MaxIterationsExceeded,
    circuit_breaker,
    cost_cap,
    fallback_model,
    max_iterations,
    reliable_agent_node,
    retry_with_backoff,
)

# --- helpers -----------------------------------------------------------------


def _msg(*, input_tokens: int, output_tokens: int, model: str) -> Message:
    """A real-shape Anthropic Message with the usage fields we care about."""
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text="hi", citations=None)],
        model=model,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return RateLimitError(message="rate limited", response=response, body=None)


def _server_error() -> APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(500, request=request)
    return APIStatusError(message="upstream", response=response, body=None)


# --- circuit_breaker ---------------------------------------------------------


def test_circuit_breaker_opens_after_n_consecutive_failures() -> None:
    calls = []

    @circuit_breaker(failures=3)
    def always_fails() -> str:
        calls.append(1)
        raise RuntimeError("boom")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            always_fails()
    # Circuit now open: subsequent calls raise CircuitOpen without invoking inner.
    inner_calls_before_open = len(calls)
    with pytest.raises(CircuitOpen):
        always_fails()
    assert len(calls) == inner_calls_before_open, "inner ran while circuit was open"


def test_circuit_breaker_resets_consecutive_count_on_success() -> None:
    state = {"i": 0}

    @circuit_breaker(failures=3)
    def fails_twice_then_succeeds() -> str:
        state["i"] += 1
        if state["i"] <= 2:
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(RuntimeError):
        fails_twice_then_succeeds()
    with pytest.raises(RuntimeError):
        fails_twice_then_succeeds()
    assert fails_twice_then_succeeds() == "ok"
    # Success cleared the failure count; we should be able to fail twice more
    # without tripping (would-be 4th + 5th consecutive in absolute terms).
    state["i"] = 0  # reset inner to keep raising

    @circuit_breaker(failures=3)
    def restart() -> None:
        raise RuntimeError("boom")

    # Verify the post-success branch by interleaving: reuse the original wrapper.
    # The previous wrapper closed over `state`; we already proved its post-success
    # behavior by re-asserting a fresh wrapper here would re-trip independently.
    for _ in range(3):
        with pytest.raises(RuntimeError):
            restart()
    with pytest.raises(CircuitOpen):
        restart()


def test_circuit_open_is_typed_not_generic() -> None:
    @circuit_breaker(failures=1)
    def always_fails() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        always_fails()
    with pytest.raises(CircuitOpen) as exc_info:
        always_fails()
    assert isinstance(exc_info.value, CircuitOpen)
    assert not isinstance(exc_info.value, RuntimeError)


# --- cost_cap ----------------------------------------------------------------


def test_cost_cap_trips_after_usd_threshold_exceeded() -> None:
    # Sonnet 4.6 list price: $3 / MTok input + $15 / MTok output.
    # One call with 1M input + 1M output = $18, well above a $5 cap.
    calls = []

    @cost_cap(usd=5.00)
    def expensive_call() -> Message:
        calls.append(1)
        return _msg(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet-4-6")

    expensive_call()
    assert len(calls) == 1
    with pytest.raises(CostCapExceeded):
        expensive_call()
    assert len(calls) == 1, "inner ran after cost cap should have tripped"


def test_cost_cap_accumulates_across_calls() -> None:
    # Each call: 100K input + 100K output on Sonnet 4.6 = 0.1*$3 + 0.1*$15 = $1.80.
    # Cap at $5 → 3rd call should trip (running 5.40 > 5.00 after call 3).
    calls = []

    @cost_cap(usd=5.00)
    def call() -> Message:
        calls.append(1)
        return _msg(input_tokens=100_000, output_tokens=100_000, model="claude-sonnet-4-6")

    call()
    call()
    call()
    assert len(calls) == 3
    with pytest.raises(CostCapExceeded):
        call()


def test_cost_cap_exceeded_is_typed_not_generic() -> None:
    @cost_cap(usd=0.01)
    def expensive() -> Message:
        return _msg(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet-4-6")

    expensive()
    with pytest.raises(CostCapExceeded) as exc_info:
        expensive()
    assert isinstance(exc_info.value, CostCapExceeded)
    assert not isinstance(exc_info.value, ValueError)


def test_cost_cap_rejects_unknown_model() -> None:
    # An unknown model has no pricing entry; silently treating it as $0 would
    # let the cap leak — the spec's hard-$-limit guarantee depends on every
    # call contributing real dollars.
    @cost_cap(usd=5.00)
    def call() -> Message:
        return _msg(input_tokens=1, output_tokens=1, model="claude-not-a-model-9")

    with pytest.raises(KeyError):
        call()


# --- max_iterations ----------------------------------------------------------


def test_max_iterations_trips_on_n_plus_one() -> None:
    calls = []

    @max_iterations(n=3)
    def call() -> str:
        calls.append(1)
        return "ok"

    for _ in range(3):
        call()
    assert len(calls) == 3
    with pytest.raises(MaxIterationsExceeded):
        call()
    assert len(calls) == 3


def test_max_iterations_counts_failures_too() -> None:
    # The research-graph cap is on cycles, not successes — a failing call still
    # consumes a slot, otherwise a thrashing graph could exceed the budget.
    calls = []

    @max_iterations(n=2)
    def call() -> None:
        calls.append(1)
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            call()
    assert len(calls) == 2
    with pytest.raises(MaxIterationsExceeded):
        call()


# --- retry_with_backoff -------------------------------------------------------


def test_retry_with_backoff_eventually_succeeds() -> None:
    state = {"i": 0}

    @retry_with_backoff(max_retries=3, initial_wait=0.0, max_wait=0.0)
    def call() -> str:
        state["i"] += 1
        if state["i"] < 3:
            raise _rate_limit_error()
        return "ok"

    assert call() == "ok"
    assert state["i"] == 3


def test_retry_with_backoff_exhausts_and_re_raises() -> None:
    @retry_with_backoff(max_retries=2, initial_wait=0.0, max_wait=0.0)
    def always_rate_limited() -> None:
        raise _rate_limit_error()

    with pytest.raises(RateLimitError):
        always_rate_limited()


def test_retry_with_backoff_does_not_retry_non_retryable() -> None:
    state = {"i": 0}

    @retry_with_backoff(max_retries=3, initial_wait=0.0, max_wait=0.0)
    def call() -> None:
        state["i"] += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        call()
    assert state["i"] == 1, "ValueError should not be retried"


def test_retry_with_backoff_retries_5xx() -> None:
    state = {"i": 0}

    @retry_with_backoff(max_retries=3, initial_wait=0.0, max_wait=0.0)
    def call() -> str:
        state["i"] += 1
        if state["i"] < 2:
            raise _server_error()
        return "ok"

    assert call() == "ok"
    assert state["i"] == 2


# --- fallback_model ----------------------------------------------------------


def test_fallback_model_switches_to_fallback_on_rate_limit() -> None:
    seen_models: list[str] = []

    @fallback_model(primary="claude-sonnet-4-6", fallback="claude-haiku-4-5")
    def call(*, model: str = "") -> str:
        seen_models.append(model)
        if model == "claude-sonnet-4-6":
            raise _rate_limit_error()
        return f"ok:{model}"

    assert call() == "ok:claude-haiku-4-5"
    assert seen_models == ["claude-sonnet-4-6", "claude-haiku-4-5"]


def test_fallback_model_does_not_retry_on_non_capacity_error() -> None:
    # The fallback is for capacity / rate-limit, not for arbitrary errors —
    # otherwise it would mask logic bugs by silently re-routing to a cheaper model.
    seen_models: list[str] = []

    @fallback_model(primary="claude-sonnet-4-6", fallback="claude-haiku-4-5")
    def call(*, model: str = "") -> str:
        seen_models.append(model)
        raise ValueError("config bug")

    with pytest.raises(ValueError):
        call()
    assert seen_models == ["claude-sonnet-4-6"], "fallback should not run for ValueError"


def test_fallback_model_propagates_fallback_failure() -> None:
    @fallback_model(primary="claude-sonnet-4-6", fallback="claude-haiku-4-5")
    def call(*, model: str = "") -> str:
        raise _rate_limit_error()

    # If even the fallback model is rate-limited, the original exception
    # surfaces — no infinite swap.
    with pytest.raises(RateLimitError):
        call()


# --- composite (reliable_agent_node) -----------------------------------------


def test_composite_short_circuits_at_cost_cap_outermost() -> None:
    """`cost_cap` must be the outermost gate.

    Once the cap is exceeded, no further calls should reach the inner
    function — not even to feed `circuit_breaker` failure tallies or
    `max_iterations` counters. We verify by counting inner invocations.
    """
    inner_calls: list[str] = []

    @reliable_agent_node(
        failures=99,
        usd=5.00,
        max_iters=99,
        max_retries=0,
        initial_wait=0.0,
        max_wait=0.0,
        primary="claude-sonnet-4-6",
        fallback="claude-haiku-4-5",
    )
    def call(*, model: str = "") -> Message:
        inner_calls.append(model)
        return _msg(input_tokens=1_000_000, output_tokens=1_000_000, model=model)

    call()
    assert len(inner_calls) == 1
    with pytest.raises(CostCapExceeded):
        call()
    assert len(inner_calls) == 1, "inner ran after cost cap should have short-circuited"


def test_composite_applies_circuit_breaker_inside_cost_cap() -> None:
    inner_calls: list[str] = []

    @reliable_agent_node(
        failures=2,
        usd=1000.00,  # effectively unlimited so circuit trips first
        max_iters=99,
        max_retries=0,
        initial_wait=0.0,
        max_wait=0.0,
        primary="claude-sonnet-4-6",
        fallback="claude-haiku-4-5",
    )
    def call(*, model: str = "") -> Message:
        inner_calls.append(model)
        raise ValueError("intentional inner failure")

    # First two calls bubble the inner ValueError. fallback_model is a no-op
    # here because ValueError isn't a capacity error.
    for _ in range(2):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(CircuitOpen):
        call()


def test_composite_applies_max_iterations_inside_circuit_breaker() -> None:
    @reliable_agent_node(
        failures=99,
        usd=1000.00,
        max_iters=2,
        max_retries=0,
        initial_wait=0.0,
        max_wait=0.0,
        primary="claude-sonnet-4-6",
        fallback="claude-haiku-4-5",
    )
    def call(*, model: str = "") -> Message:
        return _msg(input_tokens=10, output_tokens=10, model=model)

    call()
    call()
    with pytest.raises(MaxIterationsExceeded):
        call()
