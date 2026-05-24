"""Reliability primitives for agent / worker LLM calls (per `docs/CONTRACTS.md` §5).

Three composable decorators plus a `@reliable_agent_node` composite:

- `@cost_cap(usd=...)` — hard per-session USD limit, read from the real
  `anthropic.types.Usage` block on every returned `Message`. Raises
  `CostCapExceeded` (typed) once the running total crosses the cap.
- `@circuit_breaker(failures=...)` — N consecutive failures stop further
  calls; raises `CircuitOpen` (typed).
- `@retry_with_backoff(max_retries=...)` — exponential backoff with jitter
  on Anthropic `RateLimitError` / 5xx `APIStatusError` and transient httpx
  network errors. Non-retryable exceptions propagate immediately.

Composite order (outermost → innermost):

    @cost_cap                 # short-circuit if the budget is already gone
    @circuit_breaker          # short-circuit if too many consecutive failures
    @retry_with_backoff       # transient retries on the innermost call
    def call(...) -> Message: ...

`cost_cap` is outermost so an exceeded budget can't be paid down further
by retries.

State is per-wrapper instance: each decorator application creates a fresh
closure. The wrappers carry an internal `threading.Lock` to protect the
read-modify-write of their counters in case nodes are dispatched
concurrently. Whether LangGraph runs nodes via threadpool (`graph.invoke`)
or asyncio (`graph.ainvoke`) is a W2 decision; `threading.Lock` is
correct for the sync case and a no-op-cost forward bet — if the first
real caller turns out to be async, swap to `asyncio.Lock` then.

Removed in this v1, vs the original CONTRACTS §5 surface:
- `@max_iterations` — the research-graph cycle cap belongs in LangGraph's
  `recursion_limit` config (set on `graph.invoke`), which counts node
  transitions across the whole traversal. A per-function decorator bounds
  the wrong dimension and duplicates a native primitive.
- `@fallback_model` — silent Sonnet → Haiku downgrade hides capacity
  signal we want to see, changes the model that produced the output
  (different quality per spec §7.3), and complicates cost accounting.
  Canonical strategy on 429 / 5xx is `retry_with_backoff` on the primary
  model; if Anthropic genuinely sheds load, that's a circuit-breaker
  concern.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

from anthropic import APIStatusError, RateLimitError
from anthropic.types import Message
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from auto_research._pricing import usd_for_message
from auto_research._transport import TRANSIENT_NETWORK_ERRORS

F = TypeVar("F", bound=Callable[..., Any])


# --- Typed trip exceptions ---------------------------------------------------
#
# Class names are mandated by `docs/CONTRACTS.md` §5: a failing `cost_cap`
# raises `CostCapExceeded`; a failing `circuit_breaker` raises `CircuitOpen`
# — both typed, not generic. Ruff's N818 ("Exception name should end in
# Error") is suppressed because the contract is the source of truth.


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


def _is_retryable(exc: BaseException) -> bool:
    """Predicate for tenacity's `retry_if_exception`.

    Why a predicate (not `retry_if_exception_type`): `RateLimitError`
    subclasses `APIStatusError`, so naming both in a type-list would
    retry *every* `APIStatusError` — including 4xx programmer errors
    (400 bad-request, 401 unauthorized) that should fail loudly, not
    backoff-loop. The predicate filters precisely to 429 + 5xx + httpx
    transport errors.
    """
    if isinstance(exc, RateLimitError):
        return True
    if _is_5xx(exc):
        return True
    return isinstance(exc, TRANSIENT_NETWORK_ERRORS)


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
            except Exception:
                # `Exception`, not `BaseException`: a user `KeyboardInterrupt`
                # or `SystemExit` mid-call shouldn't burn a circuit slot. Those
                # signal "the process is going away," not "the API is sick."
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


# --- CostTracker + @cost_cap -------------------------------------------------
#
# The decorator shape (wrap a single Message-returning call) doesn't fit every
# caller. The batch client (`extract/batch_client.py`) accumulates cost across
# N messages emitted at once by `results()` — there's no single call to
# decorate. Both callers want the same primitive: a thread-safe USD
# accumulator with a hard cap that raises `CostCapExceeded` on overage.
# `CostTracker` is that primitive; `cost_cap` is the decorator wiring for the
# Message-returning call site.


class CostTracker:
    """Thread-safe USD accumulator with a hard cap.

    Used by both the `@cost_cap` decorator (closure state) and the batch
    client's `BatchClient` (instance state). The shared implementation
    means the lock discipline, threshold check, and error message format
    are pinned in one place; the decorator and the batch client just
    differ in *when* they call `add_message` vs `check_or_raise`.

    `check_or_raise` is on the *past* spend, not a forecast: we never
    preview the next call's cost (we'd be guessing). This is what makes
    `cost_cap` a "hard ceiling on past spend" rather than a budget.
    """

    def __init__(self, *, usd_cap: float) -> None:
        if usd_cap <= 0:
            raise ValueError("`usd_cap` must be > 0")
        self._cap = usd_cap
        self._running = 0.0
        self._lock = threading.Lock()

    def check_or_raise(self, *, where: str = "") -> None:
        """Raise `CostCapExceeded` if the accumulated spend already crossed
        the cap. Safe to call before every billable operation.
        """
        with self._lock:
            running = self._running
        if running > self._cap:
            qualifier = f" on {where}" if where else ""
            raise CostCapExceeded(
                f"cost_cap exceeded{qualifier}: "
                f"${running:.4f} > ${self._cap:.2f}"
            )

    def add_message(self, message: Message) -> float:
        """Accumulate the USD cost of `message` (via `usd_for_message`) into
        the running total. Returns the delta added — convenient for
        callers that also want to emit it as telemetry without computing
        the figure twice.
        """
        delta = usd_for_message(message)
        with self._lock:
            self._running += delta
        return delta

    def reset(self) -> None:
        """Zero the running total. For tests and explicit session
        boundaries; production code should rely on per-worker cap
        enforcement rather than calling this.
        """
        with self._lock:
            self._running = 0.0

    def running_usd(self) -> float:
        with self._lock:
            return self._running


def cost_cap(*, usd: float) -> Callable[[F], F]:
    """Hard USD limit on cumulative spend for the decorated callable.

    The inner function must return an `anthropic.types.Message`. After each
    call, the response's `usage` is converted to USD via `usd_for_message`
    and added to the running total via a `CostTracker`. Once the running
    total crosses `usd`, every subsequent call raises `CostCapExceeded`
    without invoking the inner — the cap is a ceiling on past spend, not
    a budget for the current call.

    For non-Message return shapes (e.g., the batch client's
    `BatchClient.results()` emits N messages from one call), instantiate
    `CostTracker` directly instead of using this decorator.
    """
    if usd <= 0:
        raise ValueError("`usd` must be > 0")

    def decorate(func: F) -> F:
        tracker = CostTracker(usd_cap=usd)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracker.check_or_raise(where=func.__qualname__)
            result = func(*args, **kwargs)
            if isinstance(result, Message):
                tracker.add_message(result)
            return result

        # Expose the tracker's introspection / reset methods on the
        # wrapper for parity with the pre-CostTracker API. Production
        # code rarely uses these; tests do.
        wrapper.reset = tracker.reset  # type: ignore[attr-defined]
        wrapper.running_usd = tracker.running_usd  # type: ignore[attr-defined]
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

    Uses tenacity's `retry_if_exception(_is_retryable)` predicate so 4xx
    programmer errors (400 / 401 / 404) propagate immediately instead of
    burning the backoff budget.
    """
    if max_retries < 0:
        raise ValueError("`max_retries` must be >= 0")

    def decorate(func: F) -> F:
        decorated: F = retry(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )(func)
        return decorated

    return decorate


# --- composite: @reliable_agent_node -----------------------------------------


def reliable_agent_node(
    *,
    failures: int = 3,
    usd: float = 5.00,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
) -> Callable[[F], F]:
    """Apply all three primitives in the contract-mandated order.

    Outer → inner: cost_cap, circuit_breaker, retry_with_backoff. See
    module docstring for rationale (and for the two primitives dropped vs
    the v1 CONTRACTS §5 surface).
    """

    def decorate(func: F) -> F:
        wrapped = retry_with_backoff(
            max_retries=max_retries,
            initial_wait=initial_wait,
            max_wait=max_wait,
        )(func)
        wrapped = circuit_breaker(failures=failures)(wrapped)
        wrapped = cost_cap(usd=usd)(wrapped)
        return wrapped

    return decorate


__all__ = [
    "CircuitOpen",
    "CostCapExceeded",
    "CostTracker",
    "circuit_breaker",
    "cost_cap",
    "reliable_agent_node",
    "retry_with_backoff",
]
