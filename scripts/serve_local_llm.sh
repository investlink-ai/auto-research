#!/usr/bin/env bash
# Spin up the locked local-LLM serving stack on Mac.
#
# Pinned in `learning/2026-05-28-extraction-pipeline-cost-model.md` §10.5
# "Locked stack". This is the operator-facing source of truth for the
# launch flags; the doc references this script and vice versa so the
# two don't drift.
#
# Why the flags are NOT defaults:
#
# - `--enable-prefix-cache` — the cache-economics framing depends on
#   it (cost-model doc §3.4 cache-read canary). Without it the
#   contextual chunker reprocesses the shared system prefix on every
#   call and burns most of the per-call budget.
# - `--max-request-tokens 16384` — sized for our actual workload
#   (max ~10-15K context on the RAG narrative path). The native
#   256K context would allocate KV-cache reserve we never use and
#   steal from the prefix-cache pool.
# - `--default-chat-template-kwargs '{"enable_thinking": false}'`
#   — Qwen 3.6 enables thinking mode by default; the model emits
#   prose-form chain-of-thought into `choices[0].message.content`
#   (not a separate `reasoning_content` channel). Junk for our
#   workload (structured JSON via response_format=json_schema, or
#   short text rewrites for contextual chunking).
#
# Foreground process; Ctrl-C stops it. For background serving wrap
# in `nohup ... &` or run under tmux / launchd.
#
# Override the defaults with env vars when needed:
#
#   MODEL=Qwen/Qwen3.6-4B-Instruct PORT=8001 ./scripts/serve_local_llm.sh

set -euo pipefail

VENV="${VLLM_MLX_VENV:-$HOME/.local/share/auto-research-local-inference}"
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# Idempotency: if something is already listening on the chosen port,
# assume it's our server (or someone else's) and exit clean. Spinning
# up a second instance would either fail or oversubscribe RAM.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is in use already. Existing listener:" >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 | tail -n +2
  echo "Stop it before relaunching, or pick another port with PORT=..." >&2
  exit 0
fi

# Bail loudly if the side venv is missing — the install command is
# right here so the operator doesn't have to chase the doc.
if [[ ! -x "$VENV/bin/vllm-mlx" ]]; then
  cat >&2 <<EOF
vllm-mlx venv not found at $VENV

Create it with:
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" vllm-mlx

EOF
  exit 1
fi

echo "[serve_local_llm] venv: $VENV"
echo "[serve_local_llm] model: $MODEL"
echo "[serve_local_llm] listening on: http://$HOST:$PORT"
echo

exec "$VENV/bin/vllm-mlx" serve "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --enable-prefix-cache \
  --max-tokens 4096 --max-request-tokens 16384 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
