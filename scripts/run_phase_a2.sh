#!/usr/bin/env bash
# Phase A2 (correctif). Re-fait un LoRA Whisper-small avec un LR plus
# conservatif (3e-5) et 2 epochs, parce qu'on a observé que la recette
# Chinese (LR 1e-4 / 1 epoch / rank 32) sur-échantillonne le base déjà
# fort en français : la WER repasse au-dessus du baseline.
#
# Lecture : pour un baseline Whisper déjà ≤ ~15 % WER, descendre l'LR
# d'un ordre de grandeur et ajouter une epoch est plus utile que rester
# sur la recette par défaut.
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

echo "=== train lora_small_v2 (LR 3e-5, 2 epochs) ==="
$PY -m src.train --model "$SMALL" --mode lora \
    --processed-dir "$PROC_S" --out outputs/adapters/lora_small_v2 \
    --batch 16 --grad-accum 1 --epochs 2 --lr 3e-5 \
    --lora-rank 32 --lora-alpha 64 --warmup-ratio 0.1 \
    2>&1 | tee outputs/logs/train_lora_small_v2.log

echo "=== eval lora_small_v2 ==="
$PY -m src.eval --model "$SMALL" --adapter outputs/adapters/lora_small_v2 \
    --raw-dir "$RAW_S" --tag lora_small_v2 --batch-size 16 \
    2>&1 | tee outputs/logs/eval_lora_small_v2.log

echo "PHASE A2 DONE"
