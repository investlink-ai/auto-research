---
name: bump-prompt-version
description: Use when editing any prompt template (worker prompts, extraction prompts, research-agent node prompts, live-critic prompts). Enforces INV-6 — extraction is a pure function of (raw_doc, prompt_version); changing a prompt without bumping its version invalidates the content-hash cache contract and silently corrupts eval baselines. This is the bug interviewers probe for under "how do you handle prompt evolution."
argument-hint: "[path to the prompt file you edited, if known]"
allowed-tools: Read, Grep, Bash, Edit
---

# bump-prompt-version

**Invariant being defended (AGENTS.md INV-6):**

> Extraction workers are
> `(raw_doc, prompt_version, schema_version, model_id, decoding_params) → ExtractionOutput`
> pure functions with content-hash idempotent cache. Prompt registry lives
> in Langfuse; prompt and schema versions are colocated in code. Changing
> any of the five inputs without invalidating the cache key silently
> corrupts outputs. This skill defends prompt + schema co-versioning; the
> cache key itself defends `model_id` and `decoding_params`.

The cache key is `sha256(raw_doc.bytes + prompt_version)`. If the prompt text
changes but `prompt_version` doesn't:

- Next extraction reads stale cached outputs from the **old** prompt.
- DeepEval / Ragas baselines drift because some docs reuse cache and some
  re-extract.
- The reproducibility claim ("re-running gives the same output") is false.
- MLflow runs comparing extraction quality across prompt revisions are
  invalid — they're comparing cache hits, not prompts.

This is the kind of bug that won't surface until someone asks the reproducibility
question. Defend it mechanically.

Sister skills: [[pit-check]] defends INV-1; [[citation-check]] defends INV-2.

## When to invoke

Before claiming done on any change that touches:

- Any file under `src/auto_research/extract/**/prompts/`.
- Any file matching `*_prompt.py` or `*_prompts.py`.
- A multi-line string literal inside a worker that is clearly an LLM prompt
  (heuristic: contains `{` placeholders + ends in a question or imperative).
- `src/auto_research/agents/research_graph.py` node-level prompts.
- `src/auto_research/agents/live_critic.py` Pydantic AI instructions.
- Few-shot example files under `eval/gold_sets/` when they're referenced by
  prompt templates at runtime.

## Where prompts and their versions live

Each prompt lives next to its version constant:

```python
# src/auto_research/extract/prompts/ten_k_guidance.py

TEN_K_GUIDANCE_PROMPT_VERSION = "v4"   # bump this when the prompt text changes

TEN_K_GUIDANCE_PROMPT = """\
Extract guidance-tone claims from the following 10-K MD&A section.
Every claim must include source_span (character offsets in the input)
and source_quote (verbatim slice).

<input>
{section_text}
</input>
"""
```

The version is also registered in Langfuse with `prompt_version` as the
registry tag. The registry holds the canonical history.

## What "correct" looks like

```diff
- TEN_K_GUIDANCE_PROMPT_VERSION = "v4"
+ TEN_K_GUIDANCE_PROMPT_VERSION = "v5"

  TEN_K_GUIDANCE_PROMPT = """\
- Extract guidance-tone claims from the following 10-K MD&A section.
+ Extract guidance-tone claims from the following 10-K MD&A section.
+ Include the management's confidence horizon (short / medium / long term).
  ...
  """
```

Companion steps for the same PR:

1. Push the new version to Langfuse registry:
   `uv run python -m auto_research.extract.prompts._register --name ten_k_guidance --version v5`
2. Invalidate cached entries that referenced the old version:
   `uv run python -m auto_research.extract._cache_invalidate --worker ten_k --prompt ten_k_guidance --version v4`
   (or note in the PR that the next extraction run will naturally repopulate.)
3. Re-record the DeepEval baseline on a small dev sample under the new
   version and report the delta in the PR body.

## What "wrong" looks like

```python
# WRONG — prompt edited, version unchanged
TEN_K_GUIDANCE_PROMPT_VERSION = "v4"  # unchanged
TEN_K_GUIDANCE_PROMPT = """\
Extract guidance-tone claims and ALSO classify the management's tone as ...
"""  # silently different from what v4 was when first registered

# WRONG — shared version across multiple distinct prompts
PROMPTS_VERSION = "v3"  # used by ten_k_guidance, transcript_qa, eight_k_event
# A change to one prompt now silently affects the cache key for the others.

# WRONG — version derived from prompt text hash at runtime
prompt_version = hashlib.sha256(TEN_K_GUIDANCE_PROMPT.encode()).hexdigest()[:8]
# Defeats the purpose: any whitespace change changes the version unpredictably.
# Version must be a human-readable monotonic tag for Langfuse and audit.

# WRONG — version bump without Langfuse registration
TEN_K_GUIDANCE_PROMPT_VERSION = "v5"  # bumped
# but no entry in the prompt registry → live critic can't fetch v5,
# falls back to v4 silently.

# WRONG — manual cache delete without version bump
rm -rf extract/cache/ten_k/  # next run re-extracts under SAME version
# Forces a re-run but the new outputs replace the old in MLflow tagged
# with the same prompt_version. Reproducibility broken.
```

## Mandatory checks

**1. The diff includes a version bump for every modified prompt module:**

```bash
# Replace <path> with the prompt file you edited
EDITED="src/auto_research/extract/prompts/ten_k_guidance.py"

# The same file must show a *_VERSION constant change in the diff
git diff "$EDITED" | rg -P '^[+-].*_VERSION\s*='
# Expected: exactly one - line (old version) and one + line (new version).
# Zero hits means you edited the prompt without bumping the version — STOP.
```

**2. No prompt module shares a version constant with another prompt module:**

```bash
rg -nP '^[A-Z_]+_PROMPT_VERSION\s*=' src/auto_research/extract/prompts/ \
  | awk -F: '{print $1}' | sort | uniq -c | sort -rn
# Expected: each prompt file shows up exactly once. Two prompts sharing a
# version constant (one file with two _VERSION lines, or a shared module
# import) is wrong.
```

**3. No runtime-derived version (must be a literal string, not a hash call):**

```bash
rg -nP '_PROMPT_VERSION\s*=\s*(hashlib|hash\(|sha256|os\.environ)' \
   src/auto_research/
# Expected: zero hits.
```

**4. Langfuse registry includes the new version** (run this after the
edit, before opening the PR):

```bash
uv run python -m auto_research.extract.prompts._registry_list \
  | rg -F "ten_k_guidance"
# Expected: shows v5 (or whatever you bumped to) with today's date.
# If only v4 shows, you bumped in code but didn't register — fix before PR.
```

**5. Eval baseline captured for the new version** (small dev sample is fine
for PR; full re-baseline runs on the nightly):

```bash
ls eval/baselines/ten_k_guidance__v5__*.json
# Expected: at least one baseline file exists for the new version.
```

## Pre-submit checklist

- [ ] `*_PROMPT_VERSION` constant bumped in the same file as the prompt text.
- [ ] Version is a human-readable monotonic tag (`v3` → `v4`, not a hash).
- [ ] No version constant is shared across multiple prompts.
- [ ] Langfuse registry registered with the new version.
- [ ] Cache invalidation either run explicitly or noted as deferred to the
      next nightly run.
- [ ] DeepEval / Ragas baseline captured under the new version on a dev
      sample, delta reported in the PR body.
- [ ] If this prompt feeds a worker that writes a Feast FeatureView, the
      next materialization run was noted in the PR body (since cached
      outputs feed `data/extracted/` which feeds the materializer).
- [ ] No `extract/cache/` was manually deleted under the unchanged version.

## When NOT to bump

- **Whitespace-only diffs** (auto-formatter touched the prompt string). The
  registered Langfuse prompt is the source of truth; if registry hash of the
  whitespace-normalized text didn't change, version stays. Still note the
  whitespace change in the PR.
- **Comments around the prompt, not the prompt text itself.** Same rule.
- **Test fixture prompts** under `tests/`. They're not registered in Langfuse
  and not cached against real docs.

If unsure: bump. Bumping a version unnecessarily costs a few re-extractions
on the next nightly. Not bumping when you should silently corrupts evals.

## Escalation

If a prompt edit landed on `main` without a version bump:

1. Treat as P1. Block further extraction runs.
2. Identify the affected prompt and the merge commit.
3. Compare the registered Langfuse text at the unchanged version with the
   current code — if they differ, the registry is now out-of-sync.
4. Choose recovery path:
   - **Forward fix:** bump the version *now*, register in Langfuse, accept
     that cached outputs between the offending merge and this fix used a
     hidden prompt revision. Tag the affected MLflow runs:
     `mlflow.set_tag("prompt_version_uncertain", "true")`.
   - **Roll back:** if the prompt change was small, revert it on `main` and
     reland with a proper bump.
5. Re-extract the docs touched between the offending merge and now. The
   raw store is intact; only the cache and extracted JSON need refresh.
6. Mark any DeepEval / Ragas baseline taken in that window as invalid.

The cache and `data/extracted/` are always reproducible from `data/raw/` +
the current prompt registry. The `.claude/settings.json` deny rule prevents
destructive ops on `data/raw/` — that store stays as the recovery anchor.

## Related skills

- [[citation-check]] — INV-2 defense; if a prompt edit was motivated by
  hallucination findings, both skills apply.
- [[pit-check]] — INV-1 defense; prompt-driven extraction feeds Feast, so
  cache integrity transitively affects PIT-store correctness.
