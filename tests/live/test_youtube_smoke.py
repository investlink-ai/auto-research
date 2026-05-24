"""Live smoke for the youtube source.

Hits real YouTube via yt-dlp. Confirms the search → URL → audio
download chain works end-to-end against the real backend, against
which all of our unit tests are mocked.

What this catches that unit tests + cassettes can't:

- YouTube anti-automation changes (yt-dlp warnings about JS runtimes
  signal escalating pressure; this surfaces failures before #6f's
  coverage survey hits them in bulk).
- Aggregator channel churn (Benzinga / Castify / EARNMOAR uploads
  may shift in ranking or get DMCA'd; our duration-band match
  protects against most of this, but the smoke verifies it works).
- yt-dlp version drift breaking field names (`url`, `webpage_url`,
  `duration`).

The download test is gated by `YT_DOWNLOAD_SMOKE=1` because pulling
a full earnings call is ~30-60 MB and several seconds of compute.
The discovery test runs unconditionally — it's the more brittle
half and the cheaper to exercise.
"""

from __future__ import annotations

import os

from auto_research.ingest.transcripts.sources.youtube import YouTubeSource


def test_youtube_discovers_full_length_audio_for_nvda_live() -> None:
    """Real yt-dlp search against YouTube → returns a URL whose
    duration matches the earnings-call band (30-100 min)."""
    with YouTubeSource() as src:
        url = src.find_audio_url("NVDA", 2026, 4)
    assert url is not None, "expected at least one full-length NVDA earnings call upload"
    assert url.startswith("https://"), f"expected an HTTPS URL, got {url!r}"


def test_youtube_downloads_audio_for_nvda_live() -> None:
    """End-to-end search + audio extraction. Opt-in via
    `YT_DOWNLOAD_SMOKE=1` because the download burns bandwidth."""
    if not os.environ.get("YT_DOWNLOAD_SMOKE"):
        import pytest

        pytest.skip("set YT_DOWNLOAD_SMOKE=1 to exercise the audio download path")

    with YouTubeSource() as src:
        url = src.find_audio_url("NVDA", 2026, 4)
        assert url is not None
        audio = src.download(url)
    # Earnings calls at YouTube's bestaudio AAC encoding land in the
    # ~10-80 MB range. Anything well under that is the wrong file.
    assert len(audio) > 1_000_000, f"expected meaningful audio bytes, got {len(audio)}"
