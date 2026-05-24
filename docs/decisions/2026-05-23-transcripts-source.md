# ADR: Earnings-call transcript source

- Date: 2026-05-23 (revised 2026-05-24)
- Status: **Accepted**
- Owners: Sam

## Context

Issue #6 (`feat(ingest): FMP transcript client + manifest integration`)
presupposed Financial Modeling Prep as the transcript source. Spec §3
budgeted `FMP transcripts API ~$50-100/mo`. Two investigations
invalidated that plan, in sequence:

### 1. FMP earnings transcripts are Ultimate-tier-only at $149/mo

Verified against `https://site.financialmodelingprep.com/developer/docs/pricing`.
The Ultimate tier description literally reads "everything in Premium plus
… Global Coverage Earnings Call Transcripts …" — neither Premium ($59)
nor Starter ($22) includes the endpoint. The original spec estimate
was wrong by 3×.

### 2. SEC 8-K is not a transcript source

Direct survey against `config/universe/universe_v1.json`:

- Scanned the most-recent earnings 8-K (Item 2.02 or 7.01) for every
  ticker in the 81-name universe (since reduced to 80 by removing CBRS).
- For each filing, fetched ALL text/HTML exhibits from `index.json` and
  scored each for transcript markers (`Operator:` count,
  "Question and Answer" phrase). **380 exhibits scanned in total.**
- **Result: 0/81 tickers had a transcript anywhere in 8-K exhibits.**
- Companies use 8-K Item 2.02 to file the earnings *press release*,
  never the conference-call transcript. The transcript is a separate
  artifact, posted later by third-party transcribers.

### 3. Q4 Inc is gated behind PII registration

A first design (2026-05-23) proposed using Playwright to scrape Q4
Inc's HLS player for the ~30-40% of issuer IR sites that host audio
there. Live investigation (2026-05-24) showed this path is unusable:

- `events.q4inc.com/attendee/{event_id}` pages gate the player
  behind a 5-field PII registration form. Both past and future
  events. No m3u8 request appears in network traffic before
  registration completes. Automating registration with synthetic
  identity is ToS-violating and operationally fragile.
- Search engines (Bing / DDG) index `q4inc.com` widget assets but
  not the gated `/attendee/` pages — external coverage measurement
  is also blocked.

### 4. YouTube via yt-dlp covers the universe

A universe-wide probe with yt-dlp + company-name framing found a
full-length earnings call (40-90 min duration, title containing the
SEC-canonical company name and quarter) for **every** universe
ticker. Reliable uploaders include Benzinga, Castify Earnings Call,
EARNMOAR, Investing 101, Yahoo Finance. Latency from call-end to
YouTube-availability is same-day to +1 day for the large-caps
measured.

### Paths evaluated

| Path | Year-1 cost | Engineering | Universe coverage | Legal | Notes |
|---|---|---|---|---|---|
| FMP Ultimate | $1,788 | 1-2 days | ~95% | Clean (licensed) | Spec budget overrun |
| Seeking Alpha scrape | $360-720 (Apify) | low | wide | ToS violation | Rejected |
| API Ninjas earnings_transcript | ~$120 | low | uncertain micro-cap | Clean | Coverage risk |
| Q4 Inc via Playwright | low ongoing | 6-9 days | ~30-40% (claimed) | ToS-violating (registration) | Rejected — unreachable |
| **Whisper + YouTube (yt-dlp) + per-issuer IR scrape** | **~$260 backfill + $120 ongoing = $380** | ~3-4 days actual | **80/80 measured** | Gray-zone for commercial; fine for research | **Chosen** |
| Defer entirely | $0 | 0 days | 0% | n/a | Loses signal A2 |

## Decision

Build a Whisper + YouTube + per-issuer IR-scrape pipeline:

- **`direct_mp3` source** — for tickers whose IR site hosts a plain
  MP3/M4A on a stable URL. Currently zero registry entries (the
  static probe found a few candidates but Cloudflare blocked the
  most promising). The class is the reference implementation of the
  `AudioSource` Protocol; the coverage-survey worker can populate
  entries as it discovers them.
- **`youtube` source** — primary coverage path. yt-dlp searches
  with a per-ticker company-name query, filters by 30-100 min
  duration band AND a title-gate that requires company + quarter +
  year framing all appear in the title. Audio bytes are
  magic-byte-validated before persistence. Empirically validated
  against SEC's `company_tickers.json` to catch wrong-company
  matches (see `sources/youtube.py:TICKER_QUERIES` docstring).
- **Tier-3 platforms** (Chorus Call, Brainshark, KVGO, Notified)
  remain explicitly out of scope. Tickers using only those get
  `status="no_coverage"`. The coverage survey shows zero universe
  tickers fall there today (YouTube aggregator coverage is
  complete for v1).
- **Q4 Inc** is in the same "no viable auth path" bucket as the
  tier-3 platforms. Revisit only if a non-PII access mechanism
  appears (none known today).
- **Per-ticker config** lives in `config/transcripts/sources.toml` —
  one row per universe ticker, loaded via Pydantic. The
  coverage-survey worker rewrites this file; no Python edits needed
  to add a ticker or override a query.

Commercial-use note: yt-dlp against YouTube has known ToS exposure
for commercial deployment. This project's intent (research /
paper-trading / interview portfolio) sets the risk floor; a future
commercial productization would need licensed feeds (FMP Ultimate,
AlphaSense) or direct-from-issuer paths.

## Architecture

```
src/auto_research/ingest/transcripts/
    __init__.py          # public: fetch_transcript(ticker, year, q, …)
    _base.py             # AudioSource Protocol, Transcript model, ConfigError
    _config.py           # TOML loader → ticker → source config
    _whisper.py          # OpenAI Whisper engine (chunking, retries via SDK)
    registry.py          # KNOWN_SOURCES + REGISTRY (loaded from config)
    sources/
        direct_mp3.py    # tier-1: raw MP3 URL on IR page
        youtube.py       # tier-1 in practice: yt-dlp + title-gate
        # tier-3 platforms NOT implemented; tickers fall to no_coverage

config/transcripts/sources.toml   # ticker → source + per-source overrides
config/universe/universe_v1.json  # ticker universe (80 names in v1)
```

Reuses existing primitives from issue #5: `_http.atomic_write_bytes`,
`manifest.append`, the live-smoke conftest, the rate-limiter pattern.

`Transcript` model fields per Issue #6 AC: `ticker`, `year`,
`quarter`, `event_datetime`, `prepared_remarks`, `q_and_a`.

V1 has no speaker diarization. The Q&A boundary is detected by string
match on "Question-and-Answer Session" / "We will now begin the Q&A"
in the Whisper output. If signal A2's IC is meaningfully better with
diarized speakers, a v2 PR adds `pyannote.audio`.

## Acceptance criteria status

Issue #6's original AC list:

| AC | Status |
|---|---|
| Returns frozen `Transcript` with required fields | ✅ |
| Missing coverage → `None` + manifest `no_coverage` row | ✅ |
| Manifest entries idempotent (rerun is no-op) | ✅ |
| Cassette test covers one populated transcript and one gap case | **Replaced.** The AC presupposed FMP's REST shape, which VCR records cleanly. yt-dlp's multi-hop HLS-fetch + Whisper API flow is not VCR-friendly (huge cassettes, brittle replay). The equivalent assurance is the live-smoke test (`tests/live/test_youtube_smoke.py`) + the per-source unit tests (114 in transcripts/, all using injected fakes). |

## Consequences

- Signal A2 develops against measured 100% coverage of the v1
  universe on YouTube. The IC-weighted combiner naturally weights
  down any future ticker where coverage degrades (DMCA takedowns,
  channel deletions).
- yt-dlp version drift is a real operational risk; the source pins
  `yt-dlp>=2026.3` and the live-smoke catches breakage nightly. A
  future yt-dlp major version change would surface in CI before
  user-visible failure.
- DMCA tail risk on aggregator uploads is per-quarter, not
  permanent: a takedown removes one upload while the others (4+
  uploaders per call) remain.
- Re-evaluate this ADR in W3 if Signal A2's measured IC drops
  meaningfully below T1_GATE. Mitigation paths in priority order:
  (a) populate `direct_mp3.TICKER_URL_TEMPLATES` for tickers
  served directly by their IR site (reduces YouTube dependency),
  (b) buy FMP Ultimate as a backstop, (c) drop A2 to the
  spec's §19 Cuts list.

## Survey artifacts

- `scripts/survey_8k_transcripts.py` — primary-document scan (kept)
- `scripts/survey_8k_exhibits.py` — exhaustive exhibit scan, 380 docs (kept)
- `scripts/survey_8k_transcripts.csv` / `.md` — primary-doc results
- `scripts/survey_8k_exhibits.csv` / `.md` — exhibit results

IR-audio probe scripts (`survey_ir_audio*.py`) were deleted at the time of the
Q4 Inc investigation: incomplete and superseded by the universe-wide
yt-dlp probe documented above.
