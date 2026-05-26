"""Hermetic unit tests for `extract.entity_resolution`.

LLM is mocked; embeddings use the in-process BGE backend warmed by the
session fixture in `tests/unit/conftest.py`. A 3-ticker fixture universe
keeps each test under a second.

Coverage:
- happy path: explicit name in mention -> correct ticker.
- LLM returns null -> `EntityResolution.resolved_ticker is None`.
- LLM returns off-list ticker -> downgraded to unknown with bug noted.
- LLM returns malformed JSON -> downgraded to unknown.
- Empty mention text short-circuits before the LLM call.
- Constructor rejects a universe with no aliases.
- top_k controls candidate-slate size.
- Markdown fences in the LLM response are stripped.
- `confidence` is forced to None when ticker is None even if the LLM
  contradicts itself.
"""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from pydantic import ValidationError

from auto_research.extract.embeddings import EmbeddingAdapter
from auto_research.extract.entity_resolution import (
    CandidateTicker,
    EntityResolution,
    EntityResolver,
)
from auto_research.universe import TickerEntry


def _entry(ticker: str, sector: str, aliases: tuple[str, ...]) -> TickerEntry:
    return TickerEntry(
        ticker=ticker,
        sub_universe="ai_infra",
        sector=sector,
        market_cap_tier="large",
        tradeable=True,
        aliases=aliases,
    )


def _fixture_universe() -> tuple[TickerEntry, ...]:
    return (
        _entry("NVDA", "semiconductors", ("NVIDIA", "Nvidia", "NVIDIA Corporation")),
        _entry("AMD", "semiconductors", ("AMD", "Advanced Micro Devices")),
        _entry("TSM", "semiconductors", ("TSMC", "Taiwan Semiconductor")),
    )


def _make_response(text: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=text, citations=None)],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
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


def _fake_client(text: str) -> anthropic.Anthropic:
    fake = MagicMock()
    fake.messages.create.return_value = _make_response(text)
    return cast(anthropic.Anthropic, fake)


def _bge_adapter() -> EmbeddingAdapter:
    return EmbeddingAdapter(backend="bge")


def _resolver(
    *,
    response_json: dict[str, Any] | str,
    top_k: int = 3,
    universe: tuple[TickerEntry, ...] | None = None,
) -> EntityResolver:
    text = (
        response_json if isinstance(response_json, str) else json.dumps(response_json)
    )
    return EntityResolver(
        adapter=_bge_adapter(),
        universe=universe if universe is not None else _fixture_universe(),
        anthropic_client=_fake_client(text),
        top_k=top_k,
    )


def test_resolve_returns_ticker_for_explicit_name() -> None:
    resolver = _resolver(
        response_json={
            "ticker": "NVDA",
            "confidence": 0.95,
            "reasoning": "Mention names NVIDIA H100 GPUs explicitly.",
        }
    )
    result = resolver.resolve("NVIDIA H100 GPUs")
    assert isinstance(result, EntityResolution)
    assert result.resolved_ticker == "NVDA"
    assert result.confidence == pytest.approx(0.95)
    assert "NVIDIA" in result.reasoning
    assert {c.ticker for c in result.considered} >= {"NVDA"}


def test_resolve_returns_unknown_when_llm_picks_null() -> None:
    resolver = _resolver(
        response_json={
            "ticker": None,
            "confidence": None,
            "reasoning": "Mention is too generic to disambiguate among candidates.",
        }
    )
    result = resolver.resolve("a leading semiconductor supplier")
    assert result.resolved_ticker is None
    assert result.confidence is None


def test_resolve_downgrades_off_list_ticker_to_unknown() -> None:
    # The LLM picks INTC, which isn't in the fixture universe at all.
    resolver = _resolver(
        response_json={
            "ticker": "INTC",
            "confidence": 0.8,
            "reasoning": "Mention sounds like Intel.",
        }
    )
    result = resolver.resolve("our x86 server CPU partner")
    assert result.resolved_ticker is None
    assert result.confidence is None
    assert "INTC" in result.reasoning
    assert "off-list" in result.reasoning.lower() or "not in" in result.reasoning.lower()


def test_resolve_downgrades_malformed_json_to_unknown() -> None:
    resolver = _resolver(response_json="not valid json at all")
    result = resolver.resolve("NVIDIA H100 GPUs")
    assert result.resolved_ticker is None
    assert result.confidence is None
    assert "could not be parsed" in result.reasoning.lower() or "malformed" in result.reasoning.lower()


def test_resolve_short_circuits_empty_mention() -> None:
    fake = MagicMock()
    # If the resolver were to call the LLM here, the mock would raise on
    # an unexpected attribute access — we assert messages.create was never
    # called instead, to keep the failure mode crisp.
    resolver = EntityResolver(
        adapter=_bge_adapter(),
        universe=_fixture_universe(),
        anthropic_client=cast(anthropic.Anthropic, fake),
    )
    result = resolver.resolve("   ")
    assert result.resolved_ticker is None
    assert result.considered == ()
    fake.messages.create.assert_not_called()


def test_constructor_rejects_universe_with_no_aliases() -> None:
    empty = (_entry("FOO", "semiconductors", ()),)
    with pytest.raises(ValueError, match="aliases"):
        EntityResolver(
            adapter=_bge_adapter(),
            universe=empty,
            anthropic_client=_fake_client('{"ticker": null, "confidence": null, "reasoning": "n/a"}'),
        )


def test_top_k_controls_candidate_slate_size() -> None:
    resolver = _resolver(
        response_json={
            "ticker": "NVDA",
            "confidence": 0.9,
            "reasoning": "x",
        },
        top_k=2,
    )
    result = resolver.resolve("NVIDIA chips")
    assert len(result.considered) == 2


def test_markdown_fence_stripped_from_response() -> None:
    fenced = "```json\n" + json.dumps(
        {"ticker": "NVDA", "confidence": 0.9, "reasoning": "fenced response"}
    ) + "\n```"
    resolver = _resolver(response_json=fenced)
    result = resolver.resolve("NVIDIA H100")
    assert result.resolved_ticker == "NVDA"


def test_confidence_forced_none_when_ticker_none_even_if_llm_contradicts() -> None:
    # LLM returns ticker=null but confidence=0.4 (self-contradiction). The
    # resolver normalizes this — a null ticker must carry null confidence so
    # downstream consumers can't accidentally treat 0.4 as a real score.
    resolver = _resolver(
        response_json={
            "ticker": None,
            "confidence": 0.4,
            "reasoning": "weak signal",
        }
    )
    result = resolver.resolve("a vague hyperscaler")
    assert result.resolved_ticker is None
    assert result.confidence is None


def test_universe_size_returns_distinct_ticker_count() -> None:
    resolver = _resolver(
        response_json={"ticker": "NVDA", "confidence": 0.9, "reasoning": "x"}
    )
    assert resolver.universe_size == 3


def test_top_candidates_dedupes_by_ticker() -> None:
    # NVDA has three aliases in the fixture — all three embed-rows must
    # collapse into a single NVDA candidate when ranked.
    resolver = _resolver(
        response_json={"ticker": "NVDA", "confidence": 0.9, "reasoning": "x"}
    )
    result = resolver.resolve("NVIDIA GPUs")
    tickers = [c.ticker for c in result.considered]
    assert len(tickers) == len(set(tickers))


def test_candidate_carries_primary_name_and_sector() -> None:
    resolver = _resolver(
        response_json={"ticker": "NVDA", "confidence": 0.9, "reasoning": "x"}
    )
    result = resolver.resolve("NVIDIA")
    nvda_candidate = next(c for c in result.considered if c.ticker == "NVDA")
    assert nvda_candidate.primary_name == "NVIDIA"  # first alias in fixture
    assert nvda_candidate.sector == "semiconductors"
    assert -1.0 <= nvda_candidate.score <= 1.0


def test_resolution_records_prompt_and_embed_versions() -> None:
    resolver = _resolver(
        response_json={"ticker": "NVDA", "confidence": 0.9, "reasoning": "x"}
    )
    result = resolver.resolve("NVIDIA")
    assert result.prompt_version  # non-empty string
    assert result.embed_model_version.startswith("bge:")


def test_resolution_is_frozen() -> None:
    resolver = _resolver(
        response_json={"ticker": "NVDA", "confidence": 0.9, "reasoning": "x"}
    )
    result = resolver.resolve("NVIDIA")
    with pytest.raises(ValidationError):
        result.resolved_ticker = "AMD"


def test_candidate_score_clamped_in_valid_range() -> None:
    """L2-normalized cosine is in [-1, 1]; bookkeeping should never produce a
    score outside that range. Smoke-test the constraint by inspecting the
    constructed candidates against the fixture universe."""
    resolver = _resolver(
        response_json={"ticker": "NVDA", "confidence": 0.9, "reasoning": "x"},
        top_k=3,
    )
    result = resolver.resolve("NVIDIA")
    assert all(-1.0 <= c.score <= 1.0 for c in result.considered)


def test_candidate_ticker_validates_score_range() -> None:
    # Direct schema-level check that out-of-range scores are rejected.
    with pytest.raises(ValidationError):
        CandidateTicker(
            ticker="NVDA", primary_name="NVIDIA", sector="semiconductors", score=1.5
        )
