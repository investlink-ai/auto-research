"""Reliability primitives for agent / worker LLM calls (per `docs/CONTRACTS.md` §5).

Five composable decorators plus a `@reliable_agent_node` composite:

- `@cost_cap(usd=...)` — hard per-session USD limit, read from the real
  `anthropic.types.Usage` block on every returned `Message`. Raises
  `CostCapExceeded` (typed) once the running total crosses the cap.
- `@circuit_breaker(failures=...)` — N consecutive failures stop further
  calls; raises `CircuitOpen` (typed).
- `@max_iterations(n=...)` — research-graph cycle cap; raises
  `MaxIterationsExceeded` (typed). Counts both successes and failures so
  a thrashing graph can't exceed the budget.
- `@retry_with_backoff(max_retries=...)` — exponential backoff with jitter
  on Anthropic `RateLimitError` / 5xx `APIStatusError` and transient httpx
  network errors. Non-retryable exceptions propagate immediately.
- `@fallback_model(primary=..., fallback=...)` — on a capacity-class
  exception (`RateLimitError` or HTTP 529 overload), swap the `model`
  kwarg once and re-call. No infinite swap.

Composite order (outermost → innermost):

    @cost_cap                 # short-circuit if the budget is already gone
    @circuit_breaker          # short-circuit if too many consecutive failures
    @max_iterations           # count cycles before robustness kicks in
    @retry_with_backoff       # transient retries inside one logical call
    @fallback_model           # innermost: actually invokes the API
    def call(*, model: str) -> Message: ...

`cost_cap` is outermost so an exceeded budget can't be paid down further
by retries or fallbacks. `fallback_model` is innermost so retries see the
final attempt's outcome (primary failed AND fallback failed) before
counting against `max_retries`.

State is per-wrapper instance: each decorator application creates a fresh
closure. The wrappers are thread-safe via an internal `Lock` — required
because the LangGraph research agent dispatches nodes from a thread pool.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any, Final, TypeVar, cast

import httpx
from anthropic import AnthropicError, APIStatusError, RateLimitError
from anthropic.types import Message
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

F = TypeVar("F", bound=Callable[..., Any])


# --- Typed trip exceptions ---------------------------------------------------
#
# The class names below are mandated by `docs/CONTRACTS.md` §5 and Issue #8's
# acceptance criteria ("Failing cost_cap raises CostCapExceeded; failing
# circuit_breaker raises CircuitOpen — both are typed, not generic.").
# Ruff's N818 ("Exception name should end in Error") is suppressed because the
# contract is the source of truth.


class CircuitOpen(Exception):  # noqa: N818  # contract-mandated name (CONTRACTS §5)
    """Raised when `@circuit_breaker` short-circuits a call.

    Distinct from the underlying exception type (`RuntimeError`,
    `RateLimitError`, ...) so callers — the research graph in particular —
    can distinguish "the LLM API is unhealthy enough that we should stop"
    from "this single call failed for an unrelated reason."
    """


class CostCapExceeded(Exception):  # noqa: N818  # contract-mandated name (CONTRACTS §5)
    """Raised when `@cost_cap` detects the cumulative USD spend has crossed
    the configured hard limit. The graph treats this as a terminal session
    state — there is no retry or fallback that can recover spend.
    """


class MaxIterationsExceeded(Exception):  # noqa: N818  # contract-mandated name
    """Raised when `@max_iterations` is asked for an (n+1)th call. The
    propose→backtest→decide loop is bounded at 10 cycles per spec §13.2
    — exceeding the bound means the agent is thrashing and should hand
    off to a human, not silently keep spending.
    """


# --- Pricing -----------------------------------------------------------------
#
# Anthropic list prices (USD per million tokens) for the three tiers the
# spec routes to. Cached input is billed at 10% of base input per Anthropic's
# documented prompt-caching schedule; cache *writes* are billed at 125% of
# base input. We carry the full schedule so `cost_cap` accounting stays
# truthful even when prompt caching is enabled in W2.

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


def _usd_for_message(message: Message) -> float:
    """Compute the USD cost of a single `Message` response from its `usage`.

    Raises `KeyError` if the model isn't in the pricing table — silently
    treating an unknown model as $0 would let the cap leak.
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
    # `priority` tier is 2x list but the spec doesn't route to it; leaving it
    # at list price means cost_cap will *over-bill* (trip early) rather than
    # under-bill — safe direction. Revisit if W2 adopts priority routing.
    return cost


# --- Retryable exception classification --------------------------------------
#
# What counts as transient. RateLimitError (429), 5xx APIStatusError, and
# httpx transport-level errors. ValueError / TypeError / ValidationError
# are programmer mistakes — never retry them.


def _is_5xx(exc: BaseException) -> bool:
    return (
        isinstance(exc, APIStatusError)
        and not isinstance(exc, RateLimitError)
        and 500 <= exc.status_code < 600
    )


def _is_capacity_error(exc: BaseException) -> bool:
    """529 = overloaded; 429 = rate-limited. Both indicate temporary capacity
    pressure that a model swap can route around."""
    if isinstance(exc, RateLimitError):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code == 529


_HTTPX_TRANSPORT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)


# --- @circuit_breaker --------------------------------------------------------


def circuit_breaker(*, failures: int) -> Callable[[F], F]:
    """Open the circuit after `failures` consecutive inner-call failures.

    Once open, every subsequent invocation raises `CircuitOpen` without
    invoking the inner function. Any successful inner call resets the
    consecutive-failure count to zero; the circuit cannot half-open in
    this v1 — closing requires an out-of-band `reset()` call on the
    wrapper (e.g., from session boundaries).
    """
    if failures < 1:
        raise ValueError("`failures` must be >= 1")

    def decorate(func: F) -> F:
        state = {"consecutive_failures": 0, "open": False}
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:
                if state["open"]:
                    raise CircuitOpen(
                        f"circuit_breaker open on {func.__qualname__} after "
                        f"{failures} consecutive failures"
                    )
            try:
                result = func(*args, **kwargs)
            except BaseException:
                with lock:
                    state["consecutive_failures"] += 1
                    if state["consecutive_failures"] >= failures:
                        state["open"] = True
                raise
            else:
                with lock:
                    state["consecutive_failures"] = 0
                return result

        def reset() -> None:
            with lock:
                state["consecutive_failures"] = 0
                state["open"] = False

        wrapper.reset = reset  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorate


# --- @cost_cap ---------------------------------------------------------------


def cost_cap(*, usd: float) -> Callable[[F], F]:
    """Hard USD limit on cumulative spend for the decorated callable.

    The inner function must return an `anthropic.types.Message`. After each
    call, the response's `usage` is converted to USD via `_PRICING_PER_MTOK`
    and added to the running total. Once `running_total > usd`, every
    subsequent call raises `CostCapExceeded` without invoking the inner —
    *the cap is a ceiling on past spend*, not a budget for the current
    call. (The contract's "hard $ limit" guarantee depends on this:
    if we previewed the next call's cost we'd be guessing.)
    """
    if usd <= 0:
        raise ValueError("`usd` must be > 0")

    def decorate(func: F) -> F:
        state = {"running_usd": 0.0}
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:
                if state["running_usd"] > usd:
                    raise CostCapExceeded(
                        f"cost_cap exceeded on {func.__qualname__}: "
                        f"${state['running_usd']:.4f} > ${usd:.2f}"
                    )
            result = func(*args, **kwargs)
            if isinstance(result, Message):
                with lock:
                    state["running_usd"] += _usd_for_message(result)
            return result

        def reset() -> None:
            with lock:
                state["running_usd"] = 0.0

        def running_usd() -> float:
            with lock:
                return state["running_usd"]

        wrapper.reset = reset  # type: ignore[attr-defined]
        wrapper.running_usd = running_usd  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorate


# --- @max_iterations ---------------------------------------------------------


def max_iterations(*, n: int) -> Callable[[F], F]:
    """Cap the total number of invocations at `n`. The (n+1)th raises
    `MaxIterationsExceeded`. Failures consume slots — a thrashing loop
    cannot escape the bound by repeatedly erroring out.
    """
    if n < 1:
        raise ValueError("`n` must be >= 1")

    def decorate(func: F) -> F:
        state = {"calls": 0}
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:
                if state["calls"] >= n:
                    raise MaxIterationsExceeded(
                        f"max_iterations exceeded on {func.__qualname__}: "
                        f"{state['calls']} >= {n}"
                    )
                state["calls"] += 1
            return func(*args, **kwargs)

        def reset() -> None:
            with lock:
                state["calls"] = 0

        wrapper.reset = reset  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorate


# --- @retry_with_backoff -----------------------------------------------------


def retry_with_backoff(
    *,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
) -> Callable[[F], F]:
    """Retry on rate-limit / 5xx / transient httpx errors with exponential
    backoff + jitter.

    `max_retries` is the number of *additional* attempts after the first,
    matching the contract spec's "3 retries on 5xx / rate limit." So the
    total attempt budget is `max_retries + 1`.
    """
    if max_retries < 0:
        raise ValueError("`max_retries` must be >= 0")

    retryable: tuple[type[BaseException], ...] = (
        RateLimitError,
        APIStatusError,
        *_HTTPX_TRANSPORT_ERRORS,
    )

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # We pre-filter `APIStatusError` to 5xx — `retry_if_exception_type`
            # alone would also retry 4xx (e.g., bad request), which is wrong.
            retrying = Retrying(
                stop=stop_after_attempt(max_retries + 1),
                wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
                retry=retry_if_exception_type(retryable),
                reraise=True,
            )
            try:
                for attempt in retrying:
                    with attempt:
                        try:
                            return func(*args, **kwargs)
                        except APIStatusError as exc:
                            if isinstance(exc, RateLimitError) or _is_5xx(exc):
                                raise
                            # Not retryable — re-raise outside tenacity so it
                            # doesn't get caught by retry_if_exception_type.
                            raise _NonRetryable(exc) from exc
            except _NonRetryable as wrapper_exc:
                raise wrapper_exc.original from None
            except RetryError as exc:
                # `reraise=True` should make this unreachable; defensive.
                raise exc.last_attempt.exception() from None  # type: ignore[misc]
            return None  # unreachable; keeps mypy happy across the loop

        return cast(F, wrapper)

    return decorate


class _NonRetryable(Exception):  # noqa: N818  # internal sentinel, never user-facing
    """Wrap a non-retryable APIStatusError to bypass tenacity's filter."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


# --- @fallback_model ---------------------------------------------------------


def fallback_model(*, primary: str, fallback: str) -> Callable[[F], F]:
    """Swap `model=primary` → `model=fallback` once on a capacity-class
    exception (429 / 529).

    The decorated function must accept `model` as a keyword argument. The
    decorator injects it on every call. No infinite swap: if the fallback
    also raises, the exception propagates.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If the caller explicitly passes a `model`, honor it (e.g., for
            # tests). Otherwise inject the primary.
            kwargs.setdefault("model", primary)
            try:
                return func(*args, **kwargs)
            except AnthropicError as exc:
                if not _is_capacity_error(exc):
                    raise
                # One-shot swap. Don't loop — that would mask sustained
                # capacity issues from `retry_with_backoff` and
                # `circuit_breaker` upstream.
                kwargs["model"] = fallback
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorate


# --- composite: @reliable_agent_node -----------------------------------------


def reliable_agent_node(
    *,
    failures: int = 3,
    usd: float = 5.00,
    max_iters: int = 10,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    primary: str,
    fallback: str,
) -> Callable[[F], F]:
    """Apply all five primitives in the contract-mandated order.

    Outer → inner: cost_cap, circuit_breaker, max_iterations,
    retry_with_backoff, fallback_model. See module docstring for rationale.
    """

    def decorate(func: F) -> F:
        wrapped = fallback_model(primary=primary, fallback=fallback)(func)
        wrapped = retry_with_backoff(
            max_retries=max_retries,
            initial_wait=initial_wait,
            max_wait=max_wait,
        )(wrapped)
        wrapped = max_iterations(n=max_iters)(wrapped)
        wrapped = circuit_breaker(failures=failures)(wrapped)
        wrapped = cost_cap(usd=usd)(wrapped)
        return wrapped

    return decorate


__all__ = [
    "CircuitOpen",
    "CostCapExceeded",
    "MaxIterationsExceeded",
    "circuit_breaker",
    "cost_cap",
    "fallback_model",
    "max_iterations",
    "reliable_agent_node",
    "retry_with_backoff",
]
