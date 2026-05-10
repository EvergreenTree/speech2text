#!/usr/bin/env bash
# Phase A: Whisper-small baseline / LoRA / full / from-scratch on FLEURS fr_fr.
set -euo pipefail
VENV=/data/venv
PY="$VENV/bin/python"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
ROOT=/data/speech2text
cd "$ROOT"

SMALL=openai/whisper-small
PROC_S="outputs/cache/processed/openai__whisper-small/processed"
RAW_S="outputs/cache/processed/openai__whisper-small/raw"

echo "=== baseline_small ==="
$PY -m src.eval --model "$SMALL" --raw-dir "$RAW_S" --tag baseline_small --batch-size 16 \
    2>&1 | tee outputs/logs/eval_baseline_small.log

echo "=== train lora_small ==="
$PY -m src.train --model "$SMALL" --mode lora \
    --processed-dir "$PROC_S" --out outputs/adapters/lora_small \
    --batch 16 --grad-accum 1 --epochs 1 --lr 1e-4 \
    --lora-rank 32 --lora-alpha 64 \
    2>&1 | tee outputs/logs/train_lora_small.log

echo "=== eval lora_small ==="
$PY -m src.eval --model "$SMALL" --adapter outputs/adapters/lora_small \
    --raw-dir "$RAW_S" --tag lora_small --batch-size 16 \
    2>&1 | tee outputs/logs/eval_lora_small.log

echo "=== train full_small ==="
$PY -m src.train --model "$SMALL" --mode full \
    --processed-dir "$PROC_S" --out outputs/adapters/full_small \
    --batch 8 --grad-accum 2 --epochs 1 --lr 1e-5 --gradient-checkpointing \
    2>&1 | tee outputs/logs/train_full_small.log

echo "=== eval full_small ==="
$PY -m src.eval --model outputs/adapters/full_small --raw-dir "$RAW_S" \
    --tag full_small --batch-size 16 \
    2>&1 | tee outputs/logs/eval_full_small.log

echo "=== train scratch_small ==="
$PY -m src.train --model "$SMALL" --mode scratch \
    --processed-dir "$PROC_S" --out outputs/adapters/scratch_small \
    --batch 16 --grad-accum 1 --epochs 5 --lr 5e-4 --warmup-ratio 0.05 \
    --gradient-checkpointing \
    2>&1 | tee outputs/logs/train_scratch_small.log

echo "=== eval scratch_small ==="
$PY -m src.eval --model outputs/adapters/scratch_small --raw-dir "$RAW_S" \
    --tag scratch_small --batch-size 16 \
    2>&1 | tee outputs/logs/eval_scratch_small.log

echo "PHASE A DONE"
