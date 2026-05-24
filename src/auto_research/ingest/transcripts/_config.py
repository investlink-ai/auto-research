"""TOML-backed transcripts source configuration.

Loads `config/transcripts/sources.toml` and exposes typed per-ticker
config to the rest of the `transcripts/` module. One source of truth
for:

- ticker → source mapping (consumed by `registry.REGISTRY`).
- youtube per-ticker query override (consumed by
  `sources/youtube.py:TICKER_QUERIES`).
- direct_mp3 per-ticker URL template (consumed by
  `sources/direct_mp3.py:TICKER_URL_TEMPLATES`).

Why config instead of Python dicts: each ticker is *data*, not code.
The coverage-survey worker (and humans curating the registry) edit
this file as a single auditable changeset, rather than scattered
edits across multiple `*.py` modules. Pydantic validates at load
time so a typo or unknown source name fails loud at module import.

Format choice: TOML over YAML/JSON because (a) it's in stdlib via
`tomllib` (Python ≥ 3.11), no new dep; (b) inline tables give a
one-row-per-ticker layout that diffs cleanly in PR review;
(c) comments survive (unlike JSON).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TickerSourceConfig(BaseModel):
    """Per-ticker transcript-source routing entry.

    `source` names which source class handles this ticker (must be
    one of `KNOWN_SOURCES` in `registry.py`). The optional fields
    carry source-specific overrides:

    - `query`: youtube-source query override. The default search is
      `{ticker} earnings call`; supply this when the bare ticker
      query misses or false-matches.
    - `url`: direct_mp3-source URL template with `{year}` and
      `{quarter}` placeholders. Required for direct_mp3 sources;
      ignored otherwise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    query: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _enforce_per_source_invariants(self) -> TickerSourceConfig:
        # direct_mp3 requires a URL template — otherwise there's no
        # way to dispatch a fetch. Catch this at config-load rather
        # than as a `find_audio_url` returning None mid-fetch.
        if self.source == "direct_mp3" and self.url is None:
            raise ValueError(
                "direct_mp3 source requires `url` template; got entry "
                f"with source={self.source!r} but no url"
            )
        return self


class TranscriptsSourcesConfig(BaseModel):
    """Root of `sources.toml`. Maps ticker symbol → per-ticker config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tickers: dict[str, TickerSourceConfig]


_FILENAME: Final = "sources.toml"


def _default_path() -> Path:
    """Anchor `config/transcripts/sources.toml` on the project root.

    Walks up from this module looking for `pyproject.toml` — same
    pattern as the universe loader. Editable installs (`uv sync`)
    keep the package inside the repo so this resolves
    deterministically. For non-editable installs, callers can pass
    `load_sources_config(path=...)` to override.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent / "config" / "transcripts" / _FILENAME
    raise FileNotFoundError(
        f"Could not locate the auto-research project root above {here} "
        f"(no pyproject.toml in any parent directory). Pass "
        f"`load_sources_config(path=...)` with an explicit path to "
        f"{_FILENAME}."
    )


def load_sources_config(path: Path | None = None) -> TranscriptsSourcesConfig:
    """Load + validate `config/transcripts/sources.toml`.

    Raises `FileNotFoundError` if the file is missing, `ValueError`
    for empty tables, Pydantic `ValidationError` for typed shape
    violations (unknown fields, missing `url` for direct_mp3, etc.).
    """
    target = path if path is not None else _default_path()
    with target.open("rb") as f:
        raw = tomllib.load(f)
    if not raw.get("tickers"):
        raise ValueError(
            f"{target} has no `[tickers]` table or it's empty — at least "
            "one ticker→source row is required."
        )
    return TranscriptsSourcesConfig.model_validate(raw)


__all__ = [
    "TickerSourceConfig",
    "TranscriptsSourcesConfig",
    "load_sources_config",
]
