"""Hermetic unit tests for `extract.entity_resolution`.

LLM is mocked; embeddings use the in-process BGE backend warmed by the
session fixture in `tests/unit/conftest.py`. A 3-ticker fixture universe
keeps each test under a second.
"""

from __future__ import annotations

from typing import Any, Literal, cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, ToolUseBlock, Usage
from pydantic import ValidationError

from auto_research.extract import embeddings as embeddings_module
from auto_research.extract.embeddings import EmbeddingAdapter
from auto_research.extract.entity_resolution import (
    CandidateTicker,
    EntityResolution,
    EntityResolver,
)
from auto_research.universe import TickerEntry


def _entry(ticker: str, aliases: tuple[str, ...]) -> TickerEntry:
    return TickerEntry(
        ticker=ticker,
        sub_universe="ai_infra",
        market_cap_tier="large",
        aliases=aliases,
    )


def _fixture_universe() -> tuple[TickerEntry, ...]:
    return (
        _entry("NVDA", ("NVIDIA", "Nvidia", "NVIDIA Corporation")),
        _entry("AMD", ("AMD", "Advanced Micro Devices")),
        _entry("TSM", ("TSMC", "Taiwan Semiconductor")),
    )


_StopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]


def _make_tool_response(
    *,
    tool_input: Any,
    stop_reason: _StopReason = "end_turn",
    tool_name: str = "record_extraction",
) -> Message:
    return Message(
        id="msg_test",
        content=[
            ToolUseBlock(
                id="toolu_test",
                input=tool_input,
                name=tool_name,
                type="tool_use",
            )
        ],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=100,
            output_tokens=50,
            cache_creation=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


def _make_wrong_tool_response() -> Message:
    """tool_use block emitted under a different name than `record_extraction`
    — exercises the no-record_extraction-block fallback in `resolve`."""
    return _make_tool_response(tool_input={"foo": "bar"}, tool_name="some_other_tool")


def _fake_client(message: Message | dict[str, Any]) -> anthropic.Anthropic:
    msg = (
        message
        if isinstance(message, Message)
        else _make_tool_response(tool_input=message)
    )
    fake = MagicMock()
    fake.messages.create.return_value = msg
    return cast(anthropic.Anthropic, fake)


def _bge_adapter() -> EmbeddingAdapter:
    return EmbeddingAdapter(backend="bge")


def _resolver(
    *,
    response_json: dict[str, Any] | Message,
    top_k: int = 3,
    universe: tuple[TickerEntry, ...] | None = None,
) -> EntityResolver:
    return EntityResolver(
        adapter=_bge_adapter(),
        universe=universe if universe is not None else _fixture_universe(),
        anthropic_client=_fake_client(response_json),
        top_k=top_k,
    )


# ---------- happy path & basic outcomes ----------


def test_resolve_returns_ticker_for_explicit_name() -> None:
    resolver = _resolver(
        response_json={
            "ticker": "NVDA",
            "reasoning": "Mention names NVIDIA H100 GPUs explicitly.",
        }
    )
    result = resolver.resolve("NVIDIA H100 GPUs")
    assert isinstance(result, EntityResolution)
    assert result.resolved_ticker == "NVDA"
    assert "NVIDIA" in result.reasoning
    assert {c.ticker for c in result.considered} >= {"NVDA"}


def test_resolve_returns_unknown_when_llm_picks_null() -> None:
    resolver = _resolver(
        response_json={
            "ticker": None,
            "reasoning": "Mention is too generic to disambiguate among candidates.",
        }
    )
    result = resolver.resolve("a leading semiconductor supplier")
    assert result.resolved_ticker is None


def test_resolve_downgrades_off_list_ticker_to_unknown() -> None:
    resolver = _resolver(
        response_json={
            "ticker": "INTC",
            "reasoning": "Mention sounds like Intel.",
        }
    )
    result = resolver.resolve("our x86 server CPU partner")
    assert result.resolved_ticker is None
    assert "INTC" in result.reasoning
    assert "not in" in result.reasoning.lower()


def test_resolve_downgrades_malformed_tool_input_to_unknown() -> None:
    """A `tool_use.input` dict missing required fields fails Pydantic
    validation and collapses to unknown — the equivalent of the old
    json-decode-failed branch under tool_use server-side enforcement.
    """
    resolver = _resolver(response_json={"ticker": "NVDA"})  # missing reasoning
    result = resolver.resolve("NVIDIA H100 GPUs")
    assert result.resolved_ticker is None
    assert "schema validation" in result.reasoning.lower()


def test_resolve_short_circuits_empty_mention() -> None:
    fake = MagicMock()
    resolver = EntityResolver(
        adapter=_bge_adapter(),
        universe=_fixture_universe(),
        anthropic_client=cast(anthropic.Anthropic, fake),
    )
    result = resolver.resolve("   ")
    assert result.resolved_ticker is None
    assert result.considered == ()
    fake.messages.create.assert_not_called()


# ---------- LLM-misbehavior collapse paths ----------


def test_resolve_normalizes_lowercase_ticker_to_canonical() -> None:
    """Haiku case drift ('nvda' instead of 'NVDA') must not lose a real match."""
    resolver = _resolver(
        response_json={
            "ticker": "nvda",
            "reasoning": "lowercase ticker",
        }
    )
    result = resolver.resolve("NVIDIA chips")
    assert result.resolved_ticker == "NVDA"  # canonical casing from universe


def test_resolve_normalizes_whitespace_around_ticker() -> None:
    """Trailing/leading whitespace on the LLM's ticker shouldn't trip off-list."""
    resolver = _resolver(
        response_json={
            "ticker": "  NVDA  ",
            "reasoning": "whitespace around ticker",
        }
    )
    result = resolver.resolve("NVIDIA chips")
    assert result.resolved_ticker == "NVDA"


@pytest.mark.parametrize("sentinel", ["null", "None", "n/a", "unknown", "  "])
def test_resolve_treats_stringified_nulls_as_unknown(sentinel: str) -> None:
    """LLM returning the string 'null' (or kin) is a stringification bug,
    not an off-list bug — collapse to clean unknown without the off-list
    reasoning noise.
    """
    resolver = _resolver(
        response_json={
            "ticker": sentinel,
            "reasoning": "stringified-null sentinel",
        }
    )
    result = resolver.resolve("a vague supplier")
    assert result.resolved_ticker is None
    assert "not in" not in result.reasoning.lower(), (
        f"stringified null {sentinel!r} should not surface as off-list: {result.reasoning!r}"
    )


def test_resolve_rejects_extra_fields_from_llm() -> None:
    """Pydantic extra='forbid' rejects any field outside the contract
    (ticker, reasoning) — defends against the LLM emitting a `confidence`
    field that the resolver intentionally does not accept."""
    resolver = _resolver(
        response_json={
            "ticker": "NVDA",
            "reasoning": "with extra field",
            "confidence": "high",  # not in the schema
        }
    )
    result = resolver.resolve("NVIDIA GPUs")
    assert result.resolved_ticker is None
    assert "schema validation" in result.reasoning.lower()


def test_resolve_downgrades_empty_reasoning() -> None:
    """Empty reasoning fails Pydantic min_length=1 → malformed."""
    resolver = _resolver(
        response_json={"ticker": "NVDA", "reasoning": ""}
    )
    result = resolver.resolve("NVIDIA chips")
    assert result.resolved_ticker is None
    assert result.reasoning  # the resolver's own reasoning is non-empty
    assert "schema validation" in result.reasoning.lower()


def test_resolve_handles_truncated_response_distinctly() -> None:
    """stop_reason='max_tokens' is reported as a truncation, not malformed.

    Under tool_use, a truncation surfaces as an incomplete tool_use block
    whose `input` may be partial or empty; the resolver's pre-parse
    `stop_reason` guard is what keeps that case distinguishable from
    "model emitted a wrong shape".
    """
    truncated = _make_tool_response(
        tool_input={"ticker": "NVDA"},  # truncated before reasoning
        stop_reason="max_tokens",
    )
    resolver = _resolver(response_json=truncated)
    result = resolver.resolve("NVIDIA chips")
    assert result.resolved_ticker is None
    assert "truncated" in result.reasoning.lower()
    assert "max_tokens" in result.reasoning


def test_resolve_handles_no_record_extraction_block_distinctly() -> None:
    """A response whose tool_use blocks are all under a different tool
    name collapses to unknown with a clear cause string."""
    resolver = _resolver(response_json=_make_wrong_tool_response())
    result = resolver.resolve("NVIDIA chips")
    assert result.resolved_ticker is None
    assert "no record_extraction" in result.reasoning.lower()


# ---------- construction-time guards ----------


def test_constructor_rejects_universe_with_no_aliases() -> None:
    empty = (_entry("FOO", ()),)
    with pytest.raises(ValueError, match="aliases"):
        EntityResolver(
            adapter=_bge_adapter(),
            universe=empty,
            anthropic_client=_fake_client({"ticker": None, "reasoning": "n/a"}),
        )


def test_constructor_rejects_empty_alias_string() -> None:
    """An empty alias would embed to a degenerate vector that magnets
    unrelated mentions. Fail loud at construction."""
    bad = (_entry("NVDA", ("NVIDIA", "", "Nvidia Corp")),)
    with pytest.raises(ValueError, match="empty"):
        EntityResolver(
            adapter=_bge_adapter(),
            universe=bad,
            anthropic_client=_fake_client({"ticker": None, "reasoning": "n/a"}),
        )


def test_constructor_rejects_whitespace_only_alias() -> None:
    bad = (_entry("NVDA", ("NVIDIA", "   ", "Nvidia Corp")),)
    with pytest.raises(ValueError, match="whitespace"):
        EntityResolver(
            adapter=_bge_adapter(),
            universe=bad,
            anthropic_client=_fake_client({"ticker": None, "reasoning": "n/a"}),
        )


# ---------- structural invariants ----------


def test_top_k_controls_candidate_slate_size() -> None:
    resolver = _resolver(
        response_json={"ticker": "NVDA", "reasoning": "x"},
        top_k=2,
    )
    result = resolver.resolve("NVIDIA chips")
    assert len(result.considered) == 2


def test_universe_size_returns_distinct_ticker_count() -> None:
    resolver = _resolver(response_json={"ticker": "NVDA", "reasoning": "x"})
    assert resolver.universe_size == 3


def test_top_candidates_dedupes_by_ticker() -> None:
    resolver = _resolver(response_json={"ticker": "NVDA", "reasoning": "x"})
    result = resolver.resolve("NVIDIA GPUs")
    tickers = [c.ticker for c in result.considered]
    assert len(tickers) == len(set(tickers))


def test_candidate_carries_primary_name() -> None:
    resolver = _resolver(response_json={"ticker": "NVDA", "reasoning": "x"})
    result = resolver.resolve("NVIDIA")
    nvda_candidate = next(c for c in result.considered if c.ticker == "NVDA")
    assert nvda_candidate.primary_name == "NVIDIA"  # first alias in fixture
    assert -1.0 <= nvda_candidate.score <= 1.0


def test_resolution_records_prompt_and_embed_versions() -> None:
    resolver = _resolver(response_json={"ticker": "NVDA", "reasoning": "x"})
    result = resolver.resolve("NVIDIA")
    assert result.prompt_version
    assert result.embed_model_version.startswith("bge:")


def test_embed_version_captured_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-run EMBED_MODEL_VERSION_TAG mutation must not drift the
    resolver's recorded version away from the contract under which the
    matrix vectors were produced. The resolver captures the tag at init.
    """
    monkeypatch.setattr(embeddings_module, "EMBED_MODEL_VERSION_TAG", "v1")
    resolver = _resolver(response_json={"ticker": "NVDA", "reasoning": "x"})
    monkeypatch.setattr(embeddings_module, "EMBED_MODEL_VERSION_TAG", "v2-bumped")
    result = resolver.resolve("NVIDIA")
    assert result.embed_model_version.endswith(":v1")


def test_resolution_is_frozen() -> None:
    resolver = _resolver(response_json={"ticker": "NVDA", "reasoning": "x"})
    result = resolver.resolve("NVIDIA")
    with pytest.raises(ValidationError):
        result.resolved_ticker = "AMD"


def test_candidate_score_clamped_in_valid_range() -> None:
    resolver = _resolver(
        response_json={"ticker": "NVDA", "reasoning": "x"},
        top_k=3,
    )
    result = resolver.resolve("NVIDIA")
    assert all(-1.0 <= c.score <= 1.0 for c in result.considered)


def test_candidate_ticker_validates_score_range() -> None:
    with pytest.raises(ValidationError):
        CandidateTicker(ticker="NVDA", primary_name="NVIDIA", score=1.5)


def test_top_candidates_breaks_ties_by_ticker_symbol() -> None:
    """Construct a universe where two tickers must share the top score, then
    confirm the slate order is deterministic regardless of universe load order.
    """
    universe_a = (
        _entry("ZZZA", ("identical alias",)),
        _entry("AAAA", ("identical alias",)),
    )
    universe_b = (
        _entry("AAAA", ("identical alias",)),
        _entry("ZZZA", ("identical alias",)),
    )
    r_a = _resolver(
        response_json={"ticker": "AAAA", "reasoning": "x"},
        universe=universe_a,
        top_k=2,
    )
    r_b = _resolver(
        response_json={"ticker": "AAAA", "reasoning": "x"},
        universe=universe_b,
        top_k=2,
    )
    order_a = [c.ticker for c in r_a.resolve("identical alias").considered]
    order_b = [c.ticker for c in r_b.resolve("identical alias").considered]
    assert order_a == order_b == ["AAAA", "ZZZA"]  # alphabetical tie-break


def test_returned_ticker_uses_canonical_universe_casing() -> None:
    """Even when the LLM returns a candidate ticker in mixed case, the
    EntityResolution stores the canonical (universe) casing — so downstream
    Feast lookups don't have to case-fold."""
    resolver = _resolver(
        response_json={"ticker": "Nvda", "reasoning": "mixed case"}
    )
    result = resolver.resolve("NVIDIA chips")
    assert result.resolved_ticker == "NVDA"
