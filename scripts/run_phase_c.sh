#!/usr/bin/env bash
# Phase C: chase SoTA under constraint — Whisper-large-v3-turbo + LoRA on FLEURS fr_fr.
set -euo pipefail
VENV=/data/venv
PY="$VENV/bin/python"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
ROOT=/data/speech2text
cd "$ROOT"

TURBO=openai/whisper-large-v3-turbo

$PY -m src.data --model-id "$TURBO" --n-test 500 --num-proc 4 \
    2>&1 | tee outputs/logs/dataprep_turbo.log

PROC_T="outputs/cache/processed/openai__whisper-large-v3-turbo/processed"
RAW_T="outputs/cache/processed/openai__whisper-large-v3-turbo/raw"

echo "=== baseline_turbo ==="
$PY -m src.eval --model "$TURBO" --raw-dir "$RAW_T" --tag baseline_turbo --batch-size 4 \
    2>&1 | tee outputs/logs/eval_baseline_turbo.log

echo "=== train lora_turbo ==="
# LR 5e-5 (conservative) — voir Phase A pour la justification.
$PY -m src.train --model "$TURBO" --mode lora \
    --processed-dir "$PROC_T" --out outputs/adapters/lora_turbo \
    --batch 4 --grad-accum 4 --epochs 2 --lr 5e-5 \
    --lora-rank 32 --lora-alpha 64 --gradient-checkpointing \
    2>&1 | tee outputs/logs/train_lora_turbo.log

echo "=== eval lora_turbo ==="
$PY -m src.eval --model "$TURBO" --adapter outputs/adapters/lora_turbo \
    --raw-dir "$RAW_T" --tag lora_turbo --batch-size 4 \
    2>&1 | tee outputs/logs/eval_lora_turbo.log

echo "PHASE C DONE"
