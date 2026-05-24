# ADR: Earnings-call transcript source

- Date: 2026-05-23 (updated 2026-05-24)
- Status: **Accepted** (Whisper + yt-dlp/YouTube + per-platform IR scrapers)
- Owners: Sam

## Update 2026-05-24: Q4 Inc / Playwright path abandoned, YouTube path adopted

Subsequent live investigation invalidated this ADR's tier-2 plan
(Q4 Inc via Playwright). Verified blockers:

- Q4 Inc `events.q4inc.com/attendee/{event_id}` pages gate the
  player behind a 5-field PII registration form. Both past and
  future events. No m3u8 request appears in network traffic
  before registration completes. Automating registration with
  synthetic identity is ToS-violating and operationally fragile.
- Search-engine reachability is also nil: Bing / DDG index
  `q4inc.com` widget assets but not the gated `/attendee/` pages.

In place of Q4 Inc, a direct universe-wide probe (81/81 tickers
queried via yt-dlp, company-name framing) found a full-length
in-band match on YouTube for **every** ticker. Latency from
call-end to YouTube-availability is same-day to +1 day for the
seven large-caps measured. Reliable uploaders: Benzinga, Castify
Earnings Call, EARNMOAR, Investing 101, Yahoo Finance, occasional
first-party.

**Revised rollout:**

1. **PR #6** — direct_mp3 source + Whisper + Protocol scaffold
   (shipped).
2. **PR #6-youtube** (in flight) — `sources/youtube.py` via
   yt-dlp; NVDA seeded as the canary, live-smoke against real
   YouTube. **This is the replacement for the dropped #6d.**
3. **#6f** — coverage-survey worker probes each universe ticker
   and populates `REGISTRY` + `TICKER_QUERIES` empirically;
   per-source live smoke; this ADR's "Decision" + "Architecture"
   sections get rewritten to reflect the YouTube-first reality.

Q4 Inc is now in the same "no viable auth path" bucket as the
tier-3 platforms (Chorus Call, Brainshark, KVGO, Notified) — see
spec §6.1 and the original Decision section below.

Commercial-use note: yt-dlp against YouTube has known ToS
exposure for commercial deployment. The platform's intent
(research / paper-trading) sets this risk floor; a future
commercial productization would need licensed feeds (FMP
Ultimate, AlphaSense) or direct-from-issuer paths.

The sections below are preserved verbatim from the 2026-05-23
write-up. They document the historical reasoning that produced
the (now-revised) Q4 Inc plan. **Treat the Decision and
Architecture sections as historical context**; the current
authoritative direction is this update block.

## Context

Issue #6 (`feat(ingest): FMP transcript client + manifest integration`)
presupposed Financial Modeling Prep as the transcript source. Spec §3
budgeted `FMP transcripts API ~$50-100/mo`.

Investigation revealed:

### Pricing mismatch

FMP earnings transcripts are **Ultimate-tier-only at $149/mo**, not
Premium ($59) or Starter ($22). Verified against
`https://site.financialmodelingprep.com/developer/docs/pricing`. The
Ultimate tier description literally reads "everything in Premium plus
… Global Coverage Earnings Call Transcripts …" — neither Premium nor
Starter includes the endpoint. The original spec estimate was wrong by
3×.

### SEC 8-K is empirically not a transcript source

Direct survey against `data/universe/universe_v1.json`:

- Scanned the most-recent earnings 8-K (Item 2.02 or 7.01) for every
  ticker in the 81-name universe.
- For each filing, fetched ALL text/HTML exhibits from `index.json` and
  scored each for transcript markers (`Operator:` count, "Question and
  Answer" phrase). **380 exhibits scanned in total.**
- **Result: 0/81 tickers had a transcript anywhere in 8-K exhibits.**
  Maximum `Operator:` count across the entire universe was zero.
- 8 tickers had no 8-Ks at all (foreign issuers filing 6-K: ARM, ASML,
  NVMI, SIMO, GFS, TSM, CCJ, plus pre-IPO CBRS).
- Companies use 8-K Item 2.02 to file the earnings *press release*,
  never the conference-call transcript. The transcript is a separate
  artifact, posted later by third-party transcribers.

Survey artifacts:
`scripts/survey_8k_transcripts.py`, `scripts/survey_8k_exhibits.py`,
plus CSV+MD outputs.

### IR-page static probing is insufficient

A partial probe of issuer IR pages (`scripts/survey_ir_audio.py`
deleted; conclusions captured here):

- ~10-15% of tickers expose direct MP3 / PDF transcripts in raw HTML.
- ~30-40% use Q4 Inc webcast platform (HLS streaming + JS-rendered
  player widget).
- ~15-20% use Notified / Chorus Call / KVGO / Brainshark
  (proprietary, JS-heavy).
- ~5-10% use YouTube replays.
- The rest had no detectable web audio in our static probe — modern
  React/Next.js IR sites render audio links client-side so a plain
  `httpx` probe sees an empty SSR skeleton.

A robust ingest requires headless-browser rendering (Playwright +
ffmpeg for HLS reassembly) to cover the Q4 Inc majority.

### Alternative paths evaluated

| Path | Year-1 cost | Engineering | Universe coverage | Legal | Notes |
|---|---|---|---|---|---|
| FMP Ultimate | $1,788 | 1-2 days | ~95% | Clean (licensed) | Spec budget overrun |
| Seeking Alpha scrape | $360-720 (Apify) | low | wide | ToS violation | Rejected |
| API Ninjas earnings_transcript | ~$120 | low | uncertain micro-cap | Clean | Coverage risk |
| **Whisper + Playwright + IR scrape** | **~$260 backfill + $120 ongoing = $380** | **6-9 days (with Claude Code)** | **rough estimate ~50-55%** (see Coverage caveat below) | **Clean (audio is public)** | **Chosen** |
| Defer entirely | $0 | 0 days | 0% | n/a | Loses signal A2 |

## Decision

**Build a Whisper + Playwright + per-platform IR-scrape pipeline,
shipped incrementally as PRs #6 / #6d / #6e / #6f.**

Specifically:

1. **PR #6** — `src/auto_research/ingest/transcripts/` Protocol +
   Transcript model + Whisper engine + `direct_mp3` source (tier-1
   tickers). Lands a working end-to-end for ~5-10 tickers.
2. **PR #6d** — `q4inc` source using Playwright + ffmpeg (HLS).
   Brings coverage to ~50-55% (NVDA, AAPL, MSFT, GOOGL, etc.).
3. **PR #6e** — `youtube` source via yt-dlp. Picks up a few small
   caps.
4. **PR #6f** — Full ticker→source registry, per-source live smokes,
   ADR finalization.

Tier-3 platforms (Chorus Call, Brainshark, KVGO, Notified) are
explicitly **out of scope**. Tickers using those platforms get
`status="no_coverage"` rows. Re-evaluate if signal A2 demands them.

### Coverage caveat

The "~50-55%" estimate is approximate. It came from a static-HTML
probe (`scripts/survey_ir_audio*.py` — deleted as inconclusive
because modern React/Next.js IR sites render audio links
client-side, so plain `httpx` saw empty SSR skeletons). Definitive
per-ticker tier breakdown (direct_mp3 vs q4inc vs youtube vs
unreachable) requires a Playwright-driven re-survey, which is
tracked as part of **PR #6f** (full registry + live smoke). Until
that survey lands, treat coverage numbers as design-budget rather
than measured fact.

### Engineering rationale

| Property | Whisper + Playwright | FMP Ultimate | Why we picked Whisper |
|---|---|---|---|
| Year-1 cost | $380 | $1,788 | 4.7× cheaper |
| Engineering (with Claude Code) | 6-9 days | 1-2 days | Pay engineering cost once; data cost recurs forever |
| Vendor lock-in | None | Subscription | Whisper is OpenAI's API; Q4 Inc / IR pages are owned-by-issuer |
| Latency to fresh transcript | +30-60 min | +12-48 hr | Whisper wins |
| Quality | Whisper-large-v3, near-human, no diarization in v1 | Human transcribed | Both acceptable for signal extraction |
| Coverage | ~50-55% (rest `no_coverage`) | ~95% | FMP wins coverage; we accept the tradeoff |
| Maintenance | ~1 day/quarter per platform drift | ~minimal | Whisper has a real maintenance tail |

The cost crossover happens in **month 2** of FMP subscription. Past
that point, Whisper is strictly cheaper and only the engineering
investment is in question. For a multi-quarter research project the
investment pays back many times over.

### Architecture

```
src/auto_research/ingest/transcripts/
    __init__.py          # public: fetch_transcript(ticker, year, q, …)
    _base.py             # AudioSource Protocol, Transcript model, ConfigError
    _whisper.py          # OpenAI Whisper engine (chunking, retries)
    registry.py          # ticker → source-name (data, not code)
    sources/
        direct_mp3.py    # tier-1: raw MP3 URL on IR page
        q4inc.py         # tier-2: Q4 Inc HLS via Playwright + ffmpeg
        youtube.py       # tier-2: yt-dlp wrapper
        # tier-3 explicitly NOT implemented; tickers fall through to no_coverage
```

Reuses existing primitives from #5: `_http.atomic_write_bytes`,
`manifest.append`, the live-smoke conftest, the rate-limiter pattern
(per-source `TokenBucket`).

`Transcript` model fields per Issue #6 AC:
`ticker`, `year`, `quarter`, `event_datetime`, `prepared_remarks`,
`q_and_a`.

V1 has **no speaker diarization**. The Q&A boundary is detected by
string match on "Question-and-Answer Session" / "We will now begin the
Q&A" in the Whisper output. If signal A2's IC is meaningfully better
with diarized speakers, a v2 PR adds `pyannote.audio`.

## Consequences

- Issue #6 stays open; this branch (`feat/6-transcripts`) implements
  the Protocol + Whisper + `direct_mp3` source as a first slice. Each
  follow-up (#6d/#6e/#6f) is a separate PR.
- Signal A2 develops against partial coverage. Tickers with
  `status="no_coverage"` contribute zero transcript-derived features
  to A2; the IC-weighted combiner naturally weights them down.
- Spec §6.1 (and the cost line in §3, the architecture diagram in §4,
  the workers table in §7.1, the risks list in §21) updated in this
  same change set to reflect Whisper-based ingest (no follow-up
  commit pending — the spec and ADR ship together).
- ADR is reviewed in W3 once Signal A2 has measured IC. If A2 fails
  T1_GATE because of insufficient transcript coverage, the options
  are: (a) re-include tier-3 platforms (more engineering),
  (b) subscribe to FMP Ultimate as a backstop, (c) drop A2 to the
  §19 Cuts list.

## Survey artifacts

- `scripts/survey_8k_transcripts.py` — primary-document scan (kept)
- `scripts/survey_8k_exhibits.py` — exhaustive exhibit scan, 380 docs (kept)
- `scripts/survey_8k_transcripts.csv` / `.md` — primary-doc results
- `scripts/survey_8k_exhibits.csv` / `.md` — exhibit results, definitive

IR-audio probe scripts (`survey_ir_audio*.py`) deleted: incomplete and
their conclusions captured in this ADR.
