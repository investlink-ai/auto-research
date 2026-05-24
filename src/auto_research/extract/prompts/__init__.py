"""Per-prompt modules — one file per prompt, version constant colocated.

Convention (defended by the `bump-prompt-version` skill):

- File name = prompt name (`s_filings_dilution.py`, `ten_k_guidance.py`, ...).
- Module exports two constants: `<NAME>_PROMPT_VERSION` (a `vN` tag) and
  `<NAME>_PROMPT` (the template text).
- Code is the source of truth at runtime. Langfuse is the registry —
  pushed from code via `_registry.register_prompt` for version-history
  and tag state (`dev` / `staging` / `production`).

A prompt edit that doesn't bump `*_PROMPT_VERSION` silently corrupts the
content-hash cache. The `bump-prompt-version` skill is the mechanical
guard; this docstring is the convention reference.
"""
