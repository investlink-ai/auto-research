# RUNBOOK — local LLM serving (Mac)

Operator notes for the locked vllm-mlx + Qwen 3.6-35B-A3B UD-MLX-4bit
stack that backs `make_openai_compat_extraction_client`. Architectural
rationale lives in `learning/2026-05-28-extraction-pipeline-cost-model.md`
§10.5; this file is just commands + the gotchas we hit during setup.

## First-time setup (~25 min total)

Install the side venv (kept out of the project lockfile) and
pre-download the 20 GB MLX checkpoint:

```bash
VENV=$HOME/.local/share/auto-research-local-inference
uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" vllm-mlx

# ~15 min on a typical connection; safe to interrupt + resume
"$VENV/bin/vllm-mlx" download unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit
```

## Daily ops

```bash
# Start (foreground; Ctrl-C stops it)
make serve-local-llm

# Verify reachable from another terminal
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

`make serve-local-llm` wraps `scripts/serve_local_llm.sh` — that
script is the source of truth for the load-bearing launch flags
(`--enable-prefix-cache`, `--max-request-tokens 16384`,
`--default-chat-template-kwargs '{"enable_thinking": false}'`).
Override defaults for one-offs via env vars:

```bash
MODEL=Qwen/Qwen3.6-4B-Instruct PORT=8001 make serve-local-llm
```

For background serving wrap in `nohup ... &`, tmux, or launchd.

## Post-start smoke

```python
import openai
client = openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
r = client.chat.completions.create(
    model="unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_tokens=32,
)
assert r.choices[0].finish_reason == "stop"  # not "length", not "tool_calls"
print(r.choices[0].message.content)          # short text, NO <think> tags
```

Expect ~60 tok/s sustained decode at warm cache + ~20 GB RSS on
the server process (M2 96 GB baseline). Substantially lower numbers
mean either the prefix cache is off or the model swapped to disk.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Error: --max-tokens cannot exceed --max-request-tokens` at start | Default `--max-tokens` exceeds our `--max-request-tokens 16384` cap | Use the script — it pins both to compatible values |
| Server emits prose-form chain-of-thought in `content` (`"Here's a thinking process..."`) | Qwen 3.6 thinking mode is on by default | Verify `--default-chat-template-kwargs '{"enable_thinking": false}'` reached the process: `pgrep -af vllm-mlx \| grep enable_thinking` |
| `Port 8000 is in use already` | Prior server still running, or another process owns the port | `lsof -nP -iTCP:8000 -sTCP:LISTEN` to identify, `pkill -f "vllm-mlx serve"` to stop ours, or `PORT=8001 make serve-local-llm` |
| `vllm-mlx venv not found at ...` on start | First-time setup not run on this host | Follow "First-time setup" above; the error message also quotes the install command verbatim |
| `cache_read_input_tokens` always 0 in OTel | Prefix cache off, or successive calls don't share a system prefix | Confirm `--enable-prefix-cache`; then check the workload — contextual chunking shares prefix per-parent, RAG narrative shares per-worker |
| Sustained tok/s drops below ~30 | Embedding workload contending for memory bandwidth, or another large process running | `top -o MEM` to spot competitors; consider moving the embedding model to a smaller checkpoint (Qwen3-Embedding-0.6B) per cost-model doc §10.5 |
| Decode returns `finish_reason="length"` on a short prompt | `--max-tokens 4096` cap hit; usually means the model started thinking despite the flag | Re-check the `enable_thinking` flag landed; ad-hoc: pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on the client call to force per-request |

## Stopping cleanly

Ctrl-C in the foreground terminal is the supported stop. From elsewhere:

```bash
pkill -f "vllm-mlx serve"
```

Confirm with `lsof -nP -iTCP:8000 -sTCP:LISTEN`; should return nothing.

## References

- Architecture rationale + cost / quality tradeoffs:
  `learning/2026-05-28-extraction-pipeline-cost-model.md` §9, §10
- Launch script (source of truth for flags):
  `scripts/serve_local_llm.sh`
- Wrapper this server backs:
  `src/auto_research/extract/openai_compat_client.py`
- Routing-table constants:
  `src/auto_research/_models.py` (`_LOCAL_QWEN_*`)
