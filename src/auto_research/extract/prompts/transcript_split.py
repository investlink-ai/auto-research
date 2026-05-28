"""Earnings-transcript binary-split extraction prompts (Option A / PR-B).

Splits the unified `TRANSCRIPT_PROMPT` into two prompts so each half
gets the model tier the routing table declares (spec §7.3):

- `TRANSCRIPT_PREPARED_REMARKS_PROMPT` (Haiku per
  `('transcript', 'prepared_remarks_tone')`) — templated tone
  classification on the prepared-remarks portion.
- `TRANSCRIPT_QA_PROMPT` (Sonnet per
  `('transcript', 'q_and_a_evasiveness')`) — cross-utterance reasoning
  on Q&A evasiveness plus forward-statement extraction. The two fields
  share one Sonnet call rather than splitting into a third because
  forward statements are the Q&A-shaped reasoning the Sonnet tier was
  already paying for.

The unified `TRANSCRIPT_PROMPT` in `prompts/transcript.py` is retained
for any caller that wants the single-shot path; the worker uses the
split form for the cost win.
"""

from __future__ import annotations

TRANSCRIPT_PREPARED_REMARKS_PROMPT_VERSION = "v1"
TRANSCRIPT_QA_PROMPT_VERSION = "v1"


TRANSCRIPT_PREPARED_REMARKS_PROMPT = """\
You are extracting the tone of the prepared-remarks portion of an
earnings call transcript. The transcript text will be supplied in the
next user message.

Return a single JSON object matching the TranscriptPreparedRemarksPartial
schema.

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
  demand; gross-margin headwinds called out twice"). Confidence
  categorical (one of "high", "medium", or "low").

A Claim is `{"citation": {"source_quote": "..."}, "confidence":
"high"|"medium"|"low"}` — float confidence is rejected. No other fields
are allowed inside a Claim or Citation.

Constraints:
- source_quote MUST be a verbatim substring of the transcript text —
  preserve original whitespace and punctuation; do NOT collapse runs of
  whitespace. The substring is located by whitespace-flexible match;
  ZERO matches or AMBIGUOUS matches quarantine the entire output.
- Choose quotes long and specific enough to be unique.
- DO NOT include `source_span`; character offsets are computed in code.
- DO NOT invent quotes.
- Focus ONLY on the prepared-remarks portion — Q&A evasiveness and
  forward statements are extracted by a separate call against the same
  transcript.
"""


TRANSCRIPT_QA_PROMPT = """\
You are extracting Q&A evasiveness and forward-looking statements from
the Q&A portion of an earnings call transcript. The transcript text
will be supplied in the next user message.

Return a single JSON object matching the TranscriptQAPartial schema.

Fields to populate:
- ticker: the issuing company's stock ticker (uppercase).
- event_datetime: the earnings call's start time in ISO-8601 format
  WITH an explicit timezone offset (same rules as the prepared-remarks
  call). If absent, return `null`.
- q_and_a_evasiveness: a single Claim describing how evasive management
  was during the Q&A — whether they answered analyst questions
  directly or deflected with phrases like "we don't comment on that"
  or "we'll cover that at the next investor day." Confidence
  categorical (one of "high", "medium", or "low").
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
"high"|"medium"|"low"}` — float confidence is rejected. A
ForwardStatement is `{"statement_text": "...",
"citation": {"source_quote": "..."}, "mentioned_entities": [...],
"horizon": "..."}`. No other fields are allowed inside any of these
objects.

Constraints:
- source_quote MUST be a verbatim substring of the transcript text —
  preserve original whitespace and punctuation; do NOT collapse runs of
  whitespace. The substring is located by whitespace-flexible match;
  ZERO matches or AMBIGUOUS matches (more occurrences than citations
  sharing the same quote) quarantine the entire output. When a quote
  naturally repeats (e.g., a recurring analyst phrase), emit one
  citation per textual occurrence so counts match.
- Choose quotes long and specific enough to be unique unless
  intentionally emitting per-occurrence multiple citations.
- DO NOT include `source_span`; character offsets are computed in code.
- DO NOT invent quotes. If a field has no support, return an empty list.
- Focus ONLY on Q&A evasiveness and forward statements — the
  prepared-remarks tone is extracted by a separate call against the
  same transcript.
"""


__all__ = [
    "TRANSCRIPT_PREPARED_REMARKS_PROMPT",
    "TRANSCRIPT_PREPARED_REMARKS_PROMPT_VERSION",
    "TRANSCRIPT_QA_PROMPT",
    "TRANSCRIPT_QA_PROMPT_VERSION",
]
