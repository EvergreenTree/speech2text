#!/usr/bin/env bash
# Phase B: Whisper-medium baseline + LoRA on FLEURS fr_fr.
set -euo pipefail
VENV=/data/venv
PY="$VENV/bin/python"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
ROOT=/data/speech2text
cd "$ROOT"

MEDIUM=openai/whisper-medium

# Whisper-small and Whisper-medium share the same tokenizer (51865 tokens) and
# the same feature extractor (80-bin mel, 16 kHz, 30 s window). So the
# `processed` split for small is bit-identical to what medium would produce —
# we just reuse it. (Verified empirically.) Whisper-large-v3-turbo is different
# (128-bin mel, +1 vocab token), so Phase C re-encodes.
PROC_M="outputs/cache/processed/openai__whisper-small/processed"
RAW_M="outputs/cache/processed/openai__whisper-small/raw"

echo "=== baseline_medium ==="
$PY -m src.eval --model "$MEDIUM" --raw-dir "$RAW_M" --tag baseline_medium --batch-size 8 \
    2>&1 | tee outputs/logs/eval_baseline_medium.log

echo "=== train lora_medium ==="
# LR 5e-5 (conservative) au lieu de la recette zh-CN 1e-4. La Phase A a
# montré qu'à baseline déjà fort, 1e-4 sur-entraîne ; on applique la
# leçon directement à medium.
$PY -m src.train --model "$MEDIUM" --mode lora \
    --processed-dir "$PROC_M" --out outputs/adapters/lora_medium \
    --batch 8 --grad-accum 2 --epochs 2 --lr 5e-5 \
    --lora-rank 32 --lora-alpha 64 --gradient-checkpointing \
    2>&1 | tee outputs/logs/train_lora_medium.log

echo "=== eval lora_medium ==="
$PY -m src.eval --model "$MEDIUM" --adapter outputs/adapters/lora_medium \
    --raw-dir "$RAW_M" --tag lora_medium --batch-size 8 \
    2>&1 | tee outputs/logs/eval_lora_medium.log

echo "PHASE B DONE"
