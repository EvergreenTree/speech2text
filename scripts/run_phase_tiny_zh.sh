#!/usr/bin/env bash
# Resume phase tiny zh after eval bug fix. zh data is already encoded at
# outputs/cache/processed/cv21-zh/openai__whisper-tiny/.
set -euo pipefail
VENV=/data/venv
PY="$VENV/bin/python"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
ROOT=/data/speech2text
cd "$ROOT"

TINY=openai/whisper-tiny
PROC_ZH="outputs/cache/processed/cv21-zh/openai__whisper-tiny/processed"
RAW_ZH="outputs/cache/processed/cv21-zh/openai__whisper-tiny/raw"

echo "[$(date '+%H:%M:%S')] === baseline_tiny_zh ==="
$PY -m src.eval --model "$TINY" --raw-dir "$RAW_ZH" --tag baseline_tiny_zh \
    --language zh --batch-size 32 \
    2>&1 | tee outputs/logs/eval_baseline_tiny_zh.log

echo "[$(date '+%H:%M:%S')] === train lora_tiny_zh ==="
$PY -m src.train --model "$TINY" --mode lora \
    --processed-dir "$PROC_ZH" --out outputs/adapters/lora_tiny_zh \
    --batch 32 --grad-accum 1 --epochs 1 --lr 1e-4 --language zh \
    --lora-rank 32 --lora-alpha 64 \
    2>&1 | tee outputs/logs/train_lora_tiny_zh.log

echo "[$(date '+%H:%M:%S')] === eval lora_tiny_zh ==="
$PY -m src.eval --model "$TINY" --adapter outputs/adapters/lora_tiny_zh \
    --raw-dir "$RAW_ZH" --tag lora_tiny_zh --language zh --batch-size 32 \
    2>&1 | tee outputs/logs/eval_lora_tiny_zh.log

echo "[$(date '+%H:%M:%S')] === train full_tiny_zh ==="
$PY -m src.train --model "$TINY" --mode full \
    --processed-dir "$PROC_ZH" --out outputs/adapters/full_tiny_zh \
    --batch 32 --grad-accum 1 --epochs 1 --lr 1e-5 --language zh \
    2>&1 | tee outputs/logs/train_full_tiny_zh.log

echo "[$(date '+%H:%M:%S')] === eval full_tiny_zh ==="
$PY -m src.eval --model outputs/adapters/full_tiny_zh --raw-dir "$RAW_ZH" \
    --tag full_tiny_zh --language zh --batch-size 32 \
    2>&1 | tee outputs/logs/eval_full_tiny_zh.log

echo "[$(date '+%H:%M:%S')] PHASE TINY ZH DONE"
