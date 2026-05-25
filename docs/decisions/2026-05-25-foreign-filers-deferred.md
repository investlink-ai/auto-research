# ADR: Foreign filers (20-F / 40-F) deferred from the v1 extraction pipeline

- Date: 2026-05-25
- Status: **Accepted**
- Owners: Sam

## Context

The v1 universe in `config/universe/universe_v1.json` contains 80
tickers. Seven of them file annual reports on forms other than 10-K:

| Ticker | Issuer | Country | Annual form |
|---|---|---|---|
| ASML | ASML Holding NV | Netherlands | 20-F |
| TSM | Taiwan Semiconductor Manufacturing | Taiwan | 20-F |
| ARM | ARM Holdings plc | United Kingdom | 20-F |
| NVMI | Nova Ltd | Israel | 20-F |
| SIMO | Silicon Motion Technology | Cayman / Taiwan ops | 20-F |
| GFS | GlobalFoundries Inc | Cayman / U.S. ops | 20-F |
| CCJ | Cameco Corp | Canada | 40-F |

Form 20-F is the SEC's annual-report surface for foreign private
issuers. Form 40-F is the Canadian equivalent (Multijurisdictional
Disclosure System). Neither has the same Item schema as Form 10-K:

- 20-F has 19 numbered Items in 3 Parts (vs. 23 Items in 4 Parts for
  10-K). Risk Factors live at **Item 3D**, not Item 1A. The MD&A
  equivalent is **Item 5**, not Item 7. Financial statements are
  **Item 18**, not Item 8.
- 40-F is even more minimal — Canadian issuers piggy-back on their
  domestic Annual Information Form (AIF) and MD&A, attached as
  exhibits.

The 10-K chunker (`src/auto_research/extract/chunking.py`) is
template-tuned for the 10-K Item schema. It detects Items 1A / 7 / 8
via the canonical SEC names and won't find Item 3D / Item 5 / Item 18
in a 20-F.

## Decision

**Keep the 7 foreign filers in the v1 universe. Mark them
`feature_source=False, filing_form="20-F"` (or "40-F" for CCJ).** The
extraction pipeline reads this flag and skips them at the ingest
boundary; entity resolution continues to treat them as valid mention
targets.

Schema additions to `TickerEntry` (in
`src/auto_research/universe/__init__.py`):

```python
filing_form: Literal["10-K", "20-F", "40-F"] = "10-K"
feature_source: bool = True
```

Defaults preserve backward compatibility — every existing entry
remains valid without per-row edits. Only the 7 affected rows in
`universe_v1.json` carry non-default values explicitly.

New filter on `load_universe`:

```python
load_universe(feature_source_only=True)  # for ingest / extract pipeline
load_universe(tradeable_only=True)        # for position-taking
# Compose both for the v1 pilot's extraction-and-trading set.
```

## Why keep them in the universe instead of dropping

Two roles a universe entry plays:

1. **Feature-extraction source** — its own 10-K/transcript/8-K feeds
   the chunker → extractor → Feast. The 7 foreign filers don't have
   this role in v1 (no 10-K).

2. **Entity-resolution target** — when *other* companies' narrative
   docs mention them, mention text is mapped to a ticker. The 7
   foreign filers are *especially* important for this role:

| Ticker | Mention-target role |
|---|---|
| TSM | The foundry behind NVDA/AMD/AVGO/MRVL/AAPL silicon. Mentioned by name or as "our wafer-foundry partner" in nearly every chip 10-K. |
| ASML | The only EUV-lithography vendor for leading-edge nodes. Mentioned in AMAT/LRCX/KLAC 10-Ks. |
| ARM | Architecture licensed to QCOM/AAPL/AMD/AVGO. Mentioned in mobile/embedded 10-Ks. |
| NVMI/SIMO/GFS | Niche semicap & foundry plays — supplier-mention targets in adjacent filers. |
| CCJ | Uranium supplier — mentioned in nuclear-power 10-Ks (CEG, VST, GEV, OKLO, SMR, NNE). |

Dropping them → entity resolution can't map mentions → Signal A1
(forward-tone propagation) loses real supplier cross-references →
false negatives in supplier-mention features.

## Why not parse 20-F now

A parallel `parse_20f.py` with its own schema would need:

- A 20-F-specific item whitelist (Items 1, 2, 3A-3D, 4, 4A, 5, 6, 7,
  8, 9, 10, 11, 12, 13, 14, 15, 16A-16K, 17, 18, 19)
- Bare-title map adapted to 20-F titles ("Key Information" → Item 3,
  "Operating and Financial Review" → Item 5, "Financial Information"
  → Item 8)
- A 40-F path that follows exhibit references back to the Canadian
  AIF / MD&A docs
- IFRS-aware table extraction for Item 18 (different statement
  layouts than U.S. GAAP)
- Downstream worker awareness — the existing 10-K worker's prompts
  reference 10-K Item structure; we'd need 20-F-equivalent prompts

Rough scope: ~200-400 LOC + parallel fixture set + per-form worker
prompts. Worth doing for ASML alone (one of the most signal-rich
names in the AI-infra universe), but not a v1 blocker.

## Consequences

### Positive

- The pilot's effective feature-extraction universe is 73 names —
  reachable today without writing a parallel parser.
- Entity resolution covers the full 80-name surface, so cross-doc
  signals (A1) don't lose supplier mentions.
- The schema additions are forward-compatible: when 20-F support
  lands, we flip `feature_source: false` → `true` and ship the
  parallel parser; no universe-file migration required.

### Negative / accepted tradeoffs

- ASML, TSM, ARM cannot contribute their *own* MD&A tone to Signal
  A2 (PEAD-style language drift). A2 sees a smaller universe than
  A1 by design.
- The 1-year filing cadence on 20-F is already a thinner signal
  surface than the 10-K + 10-Q + 8-K quarterly stream U.S. filers
  produce. Even with a 20-F parser, these names contribute fewer
  data points per year.

## Rollback

Trivial: remove the two new fields from `TickerEntry` and from the 7
foreign-filer rows in `universe_v1.json`. `feature_source_only`
filter becomes a no-op. The universe file remains backward-
compatible because the new fields are defaulted.

## References

- `src/auto_research/universe/__init__.py` — `TickerEntry`,
  `load_universe(feature_source_only=...)`
- `config/universe/universe_v1.json` — annotated rows for the 7
  foreign filers
- `tests/unit/test_universe.py` — `test_foreign_filers_marked_non_feature_source`,
  `test_load_universe_feature_source_only_excludes_foreign_filers`
- Form 20-F instructions: <https://www.sec.gov/files/form20-f.pdf>
- Form 40-F instructions: <https://www.sec.gov/files/form40-f.pdf>
