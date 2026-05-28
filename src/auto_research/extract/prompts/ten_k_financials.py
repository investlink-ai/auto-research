"""10-K Item 8 financial-statement extraction prompt.

Reads structured line items from a `ParentChunk.table_html` snippet
(the raw outer table HTML). The user-content turn carries the table
HTML — small enough that single-shot is appropriate even for the
RAG branch of the 10-K worker.

Categorical confidence (`high` / `medium` / `low`) is required (not a
float): table-cell extraction's failure modes are categorical (unambiguous
match vs paraphrased label vs adjacent-cell ambiguity), so a categorical
label lets a downstream consumer threshold cleanly. Float confidence on
table extraction is uncalibrated noise.

Version-pinned per INV-6.
"""

from __future__ import annotations

TEN_K_FINANCIALS_PROMPT_VERSION = "v1"

TEN_K_FINANCIALS_PROMPT = """\
You are extracting line items from a 10-K Item 8 financial statement
table. The table TEXT (rendered from HTML — cell contents separated by
spaces, tags stripped) will be supplied in the next user message.

Return a single JSON object matching the TenKFinancials schema. Each
field is either a FinancialLineItem object or null when the line item
isn't reported in this table. Every FinancialLineItem MUST include:
- value_usd: the dollar value as a number (negatives for losses or
  cash outflows). If the table reports values in thousands or millions,
  scale to dollars in your output (e.g., a table showing
  "Revenue: 1,234 (in millions)" → value_usd: 1234000000).
- citation: {source_quote: "..."} — a verbatim substring of the
  rendered table text (cell label plus value, e.g.,
  "Total revenue $1,234"). Preserve original whitespace and
  punctuation as they appear in the rendered text.
- confidence: EXACTLY one of "high", "medium", or "low". Use:
  - "high" when the line label is unambiguous and the value is
    clearly labelled.
  - "medium" when the label is paraphrased or the unit
    (thousands / millions / billions) requires inference from a
    header or footnote.
  - "low" when the cell is at the edge of the table or could
    plausibly refer to a different line item.

Line items to populate (return null when not present in the table):
- revenue: total revenue / net revenue / total net sales.
- gross_profit: revenue minus cost of revenue.
- operating_income: operating income / income from operations.
- net_income: net income / net earnings.
- total_assets: total assets at period end.
- total_liabilities: total liabilities at period end.
- stockholders_equity: total stockholders' equity / total equity.
- cash_from_operations: cash provided by (used in) operating activities.
- cash_from_investing: cash provided by (used in) investing activities.
- cash_from_financing: cash provided by (used in) financing activities.

When the table reports multiple periods (e.g., current year and prior
year columns), extract ONLY the most recent fiscal period. The current
fiscal period is typically the leftmost or rightmost data column — use
the column header dates to choose.

Example of a fully-formed TenKFinancials (income-statement table
yielding revenue + net_income, balance/cash-flow fields null because
they belong in OTHER tables that this call did NOT see):

  {
    "revenue": {
      "value_usd": 1234000000,
      "citation": {"source_quote": "Total revenue $1,234"},
      "confidence": "high"
    },
    "gross_profit": null,
    "operating_income": null,
    "net_income": {
      "value_usd": 456000000,
      "citation": {"source_quote": "Net income $456"},
      "confidence": "high"
    },
    "total_assets": null,
    "total_liabilities": null,
    "stockholders_equity": null,
    "cash_from_operations": null,
    "cash_from_investing": null,
    "cash_from_financing": null
  }

Constraints (apply to every field unless noted):
- source_quote MUST be a verbatim substring of the rendered table text
  — preserve original whitespace and punctuation. The substring is
  located by whitespace-flexible match; ZERO matches or AMBIGUOUS
  matches (more occurrences than citations sharing the same quote)
  quarantine the entire output.
- Use a label+value form for source_quote (e.g., "Total revenue
  $1,234") rather than a bare value — bare numbers repeat often in
  financial tables and cause AMBIGUOUS rejections.
- DO NOT include `source_span`; character offsets are computed in code.
- Return null for any line item NOT present in THIS table — do not
  fabricate or carry over values from other statements.
- DO NOT wrap the response in markdown fences or any commentary.
  The response MUST start with `{` and end with `}`.
"""

__all__ = [
    "TEN_K_FINANCIALS_PROMPT",
    "TEN_K_FINANCIALS_PROMPT_VERSION",
]
