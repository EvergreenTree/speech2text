#!/usr/bin/env bash
# Phase E (optionnel) : pousser plus loin la chasse SoTA — Whisper-large-v3 (32 couches
# decoder, 1.55B paramètres) + LoRA + grad ckpt aggressive. Plus lourd que turbo
# mais peut-être plus précis. À lancer seulement si Phase C / D ont confirmé qu'on
# a du temps GPU restant dans le budget 4 h.
set -euo pipefail
VENV=/data/venv
PY="$VENV/bin/python"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
ROOT=/data/speech2text
cd "$ROOT"

LARGE=openai/whisper-large-v3

$PY -m src.data --model-id "$LARGE" --n-test 500 --num-proc 4 \
    2>&1 | tee outputs/logs/dataprep_large.log

PROC_L="outputs/cache/processed/openai__whisper-large-v3/processed"
RAW_L="outputs/cache/processed/openai__whisper-large-v3/raw"

echo "=== baseline_large ==="
$PY -m src.eval --model "$LARGE" --raw-dir "$RAW_L" --tag baseline_large --batch-size 2 \
    2>&1 | tee outputs/logs/eval_baseline_large.log

# Avec un decoder de 32 couches, batch 2 + grad-accum 8 + grad ckpt =
# effective batch 16, ~25 GB peak donc tight sur L4 23 Go : on peut être
# obligé de descendre à grad-accum 16 / batch 1 selon comportement réel.
echo "=== train lora_large ==="
$PY -m src.train --model "$LARGE" --mode lora \
    --processed-dir "$PROC_L" --out outputs/adapters/lora_large \
    --batch 2 --grad-accum 8 --epochs 2 --lr 5e-5 \
    --lora-rank 16 --lora-alpha 32 --gradient-checkpointing \
    2>&1 | tee outputs/logs/train_lora_large.log

echo "=== eval lora_large ==="
$PY -m src.eval --model "$LARGE" --adapter outputs/adapters/lora_large \
    --raw-dir "$RAW_L" --tag lora_large --batch-size 2 \
    2>&1 | tee outputs/logs/eval_lora_large.log

echo "PHASE E DONE"
