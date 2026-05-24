"""Unit tests for the Langfuse prompts-registry wrapper (Issue #11)."""

from __future__ import annotations

from unittest.mock import MagicMock

from auto_research.extract.prompts._registry import (
    register_prompt,
    set_prompt_tag,
)


def test_register_prompt_calls_create_with_label_and_returns_int_version() -> None:
    """register_prompt forwards code-side string version as a Langfuse
    label (Langfuse auto-assigns its own int version) and returns the
    int version so callers can persist the str-label -> int-version
    mapping if they need to address the prompt later."""
    client = MagicMock()
    client.create_prompt.return_value = MagicMock(version=7)
    int_version = register_prompt(
        name="s_filings_dilution",
        version="v1",
        text="extract dilution from filing",
        client=client,
    )
    client.create_prompt.assert_called_once()
    kwargs = client.create_prompt.call_args.kwargs
    assert kwargs["name"] == "s_filings_dilution"
    assert kwargs["prompt"] == "extract dilution from filing"
    assert kwargs["labels"] == ["v1"]
    assert kwargs["type"] == "text"
    assert int_version == 7


def test_set_prompt_tag_looks_up_int_version_by_label() -> None:
    """set_prompt_tag must resolve the code-side label to a Langfuse int
    version before calling update_prompt — passing the raw string label
    raises pydantic ValidationError against the real SDK."""
    client = MagicMock()
    client.get_prompt.return_value = MagicMock(version=3)
    set_prompt_tag(
        name="s_filings_dilution",
        version="v1",
        tag="production",
        client=client,
    )
    # Lookup by label first
    client.get_prompt.assert_called_once_with("s_filings_dilution", label="v1")
    # Then update by the resolved int version
    client.update_prompt.assert_called_once_with(
        name="s_filings_dilution",
        version=3,
        new_labels=["production"],
    )


def test_set_prompt_tag_passes_int_not_str_to_update() -> None:
    """Regression guard for the original review finding: an int (not the
    label string) must reach update_prompt's version kwarg."""
    client = MagicMock()
    client.get_prompt.return_value = MagicMock(version=42)
    set_prompt_tag(
        name="s_filings_dilution",
        version="v9",
        tag="production",
        client=client,
    )
    assert isinstance(
        client.update_prompt.call_args.kwargs["version"], int
    ), "update_prompt must receive int, not str (Langfuse v2 typed)"
