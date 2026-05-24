"""Unit tests for the content-hash idempotent cache (Issue #11).

Defends INV-6: cache key captures the full completion config — raw_doc,
prompt_version, schema_version, model_id, decoding_params. A change to ANY
of the five must produce a fresh cache key (and thus a fresh LLM call).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_research.extract.cache import cache_key, read, write


def test_cache_key_is_stable_for_same_inputs() -> None:
    k1 = cache_key(
        raw_doc=b"hello",
        prompt_version="v1",
        schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"max_tokens": 4096},
    )
    k2 = cache_key(
        raw_doc=b"hello",
        prompt_version="v1",
        schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"max_tokens": 4096},
    )
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


@pytest.mark.parametrize(
    "field",
    ["raw_doc", "prompt_version", "schema_version", "model_id", "decoding_params"],
)
def test_cache_key_changes_when_any_completion_config_input_changes(field: str) -> None:
    """The interview-grade test: tiered routing (Haiku→Sonnet swap) must
    not silently reuse stale cache. Each of the five inputs is a cache-key
    component; flip any one and the key changes."""
    base: dict = dict(
        raw_doc=b"hello",
        prompt_version="v1",
        schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"max_tokens": 4096},
    )
    mutated: dict = {
        "raw_doc": b"hello world",
        "prompt_version": "v2",
        "schema_version": "v2",
        "model_id": "claude-sonnet-4-6",
        "decoding_params": {"max_tokens": 8192},
    }
    other = dict(base) | {field: mutated[field]}
    assert cache_key(**base) != cache_key(**other), f"changing {field} must change the cache key"


def test_decoding_params_dict_ordering_does_not_affect_key() -> None:
    """Two dicts with the same items in different insertion order must hash
    to the same key (canonical JSON serialization)."""
    k_ab = cache_key(
        raw_doc=b"x", prompt_version="v1", schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"a": 1, "b": 2},
    )
    k_ba = cache_key(
        raw_doc=b"x", prompt_version="v1", schema_version="v1",
        model_id="claude-haiku-4-5",
        decoding_params={"b": 2, "a": 1},
    )
    assert k_ab == k_ba


def test_round_trip_write_then_read(tmp_path: Path) -> None:
    key = cache_key(
        raw_doc=b"x", prompt_version="v1", schema_version="v1",
        model_id="claude-haiku-4-5", decoding_params={},
    )
    payload = {"hello": "world", "n": 42}
    write(tmp_path, "s_filings", key, payload)
    got = read(tmp_path, "s_filings", key)
    assert got == payload


def test_read_returns_none_on_miss(tmp_path: Path) -> None:
    assert read(tmp_path, "s_filings", "deadbeef" * 8) is None
