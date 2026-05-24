"""Unit tests for the Langfuse prompts-registry wrapper (Issue #11).

Workers don't fetch prompts from Langfuse at runtime — code is the source
of truth, this wrapper only pushes for visibility and flips tags for
promotion. The Langfuse client is mocked so tests stay hermetic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from auto_research.extract.prompts._registry import (
    register_prompt,
    set_prompt_tag,
)


def test_register_prompt_calls_langfuse_create_prompt() -> None:
    client = MagicMock()
    register_prompt(
        name="s_filings_dilution",
        version="v1",
        text="extract dilution from {source_text}",
        client=client,
    )
    client.create_prompt.assert_called_once()
    kwargs = client.create_prompt.call_args.kwargs
    assert kwargs["name"] == "s_filings_dilution"
    assert kwargs["prompt"] == "extract dilution from {source_text}"
    # The Langfuse v2 SDK takes labels as the version-tag mechanism; we
    # include the code version constant as a label so a registry browser
    # can match Langfuse rows to code commits.
    assert "v1" in kwargs["labels"]


def test_set_prompt_tag_promotes_existing_version() -> None:
    client = MagicMock()
    set_prompt_tag(
        name="s_filings_dilution",
        version="v1",
        tag="production",
        client=client,
    )
    client.update_prompt.assert_called_once()
    kwargs = client.update_prompt.call_args.kwargs
    assert kwargs["name"] == "s_filings_dilution"
    assert "production" in kwargs["new_labels"]
