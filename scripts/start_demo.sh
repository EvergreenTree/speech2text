#!/usr/bin/env bash
# Launch the Gradio demo. Defaults to whisper-small base + lora_small adapter.
#
# Override: SERVER_PORT, SERVER_HOST, BASE, LORA, FULL
set -euo pipefail

VENV=${VENV:-/data/venv}
ROOT=${ROOT:-/data/speech2text}
SERVER_HOST=${SERVER_HOST:-0.0.0.0}
SERVER_PORT=${SERVER_PORT:-7860}

# Default to the best run (turbo + LoRA), then medium, then small.
if [[ -z "${BASE:-}" ]]; then
  if [[ -d "$ROOT/outputs/adapters/lora_turbo" ]]; then
    BASE=openai/whisper-large-v3-turbo
    LORA=${LORA:-$ROOT/outputs/adapters/lora_turbo}
  elif [[ -d "$ROOT/outputs/adapters/lora_medium" ]]; then
    BASE=openai/whisper-medium
    LORA=${LORA:-$ROOT/outputs/adapters/lora_medium}
  else
    BASE=openai/whisper-small
    LORA=${LORA:-$ROOT/outputs/adapters/lora_small}
  fi
fi
FULL=${FULL:-}

cd "$ROOT"
export HF_HOME=$ROOT/outputs/cache

ARGS=(
  --base "$BASE"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --dtype bf16
)
if [[ -n "$FULL" ]]; then
  ARGS+=( --full "$FULL" --ft-label "Full FT" )
elif [[ -n "$LORA" && -d "$LORA" ]]; then
  ARGS+=( --lora "$LORA" --ft-label "LoRA" )
fi

echo "Launching Gradio on $SERVER_HOST:$SERVER_PORT"
echo "  base: $BASE"
echo "  ft:   ${FULL:-${LORA:-(none)}}"
exec "$VENV/bin/python" -m src.server "${ARGS[@]}"
