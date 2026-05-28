"""Earnings-transcript extraction prompt.

Single-shot. A 1-3 hour earnings call transcribes to ~25-60K tokens —
well under `SINGLE_SHOT_TOKEN_CUTOFF`. Instructions only; transcript
text goes in user content.

Version-pinned per INV-6.
"""

from __future__ import annotations

TRANSCRIPT_PROMPT_VERSION = "v1"

TRANSCRIPT_PROMPT = """\
You are extracting language signals from an earnings call transcript.
The transcript text will be supplied in the next user message.

Return a single JSON object matching the TranscriptOutput schema. Every
claim MUST include:
- source_quote: a verbatim substring of the transcript supporting the
  claim. Preserve the original whitespace exactly — do NOT collapse
  runs of whitespace. The substring will be located in the transcript
  by whitespace-flexible match; if no occurrence is found, OR if more
  than one occurrence is found, the claim is rejected and the output
  quarantined. Choose quotes long and specific enough to be unique
  in the transcript.

DO NOT include `source_span`. Character offsets are computed in code.

Fields to populate:
- ticker: the issuing company's stock ticker (uppercase).
- event_datetime: the earnings call's start time in ISO-8601 format
  WITH an explicit timezone offset (e.g., "2026-01-30T17:00:00-05:00").
  The timezone offset is MANDATORY — a naive datetime (no offset) is
  rejected by the schema. If the transcript does not state a precise
  time AND an explicit timezone (e.g., "Eastern Time", "EST", "ET",
  "Pacific Time"), return `null`. Do NOT guess the timezone from the
  company's headquarters or from training data — a wrong offset
  silently corrupts every time-windowed downstream signal.
- prepared_remarks_tone: a single Claim describing the overall tone of
  the prepared remarks portion (e.g., "cautious bullish on FY26
  demand; gross-margin headwinds called out twice"). Confidence in [0, 1].
- q_and_a_evasiveness: a single Claim describing how evasive management
  was during the Q&A — whether they answered analyst questions
  directly or deflected with phrases like "we don't comment on that"
  or "we'll cover that at the next investor day." Confidence in [0, 1].
- forward_statements: list of ForwardStatement objects, each describing
  a forward-looking claim management made. Each ForwardStatement has:
  - statement_text: the paraphrased forward statement (e.g., "expect
    FY26 revenue growth above 30%").
  - citation: {source_quote: "..."} — verbatim substring supporting
    the paraphrase.
  - mentioned_entities: list of tickers or company names referenced
    in the statement (e.g., ["NVDA", "MSFT"]). Empty list if none.
  - horizon: phrase describing the time horizon, e.g., "next quarter",
    "FY26", "long-term", "by end of 2026", "over the next 18 months".

A Claim is `{"citation": {"source_quote": "..."}, "confidence":
0.0-1.0}`. A ForwardStatement is `{"statement_text": "...",
"citation": {"source_quote": "..."}, "mentioned_entities": [...],
"horizon": "..."}`. No other fields are allowed inside any of these
objects.

If a field has no support in the transcript, return an empty list. Do
not fabricate citations.

Return ONLY the JSON object. No markdown code fences. No commentary.
"""

__all__ = [
    "TRANSCRIPT_PROMPT",
    "TRANSCRIPT_PROMPT_VERSION",
]
