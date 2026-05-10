#!/usr/bin/env bash
# Reprend la pipeline après le kill : scratch_small (epochs 5, plus court), puis
# Phase B, Phase C, et Post (A2 + D + analyse). Utilisé une fois Phase A
# baseline + LoRA + full sont déjà dans outputs/preds/.
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

echo "[$(date '+%H:%M:%S')] === train scratch_small (epochs 5) ==="
$PY -m src.train --model "$SMALL" --mode scratch \
    --processed-dir "$PROC_S" --out outputs/adapters/scratch_small \
    --batch 16 --grad-accum 1 --epochs 5 --lr 5e-4 --warmup-ratio 0.05 \
    --gradient-checkpointing \
    2>&1 | tee outputs/logs/train_scratch_small.log

echo "[$(date '+%H:%M:%S')] === eval scratch_small ==="
$PY -m src.eval --model outputs/adapters/scratch_small --raw-dir "$RAW_S" \
    --tag scratch_small --batch-size 16 \
    2>&1 | tee outputs/logs/eval_scratch_small.log

echo "[$(date '+%H:%M:%S')] === phase B ==="
bash scripts/run_phase_b.sh

echo "[$(date '+%H:%M:%S')] === phase C ==="
bash scripts/run_phase_c.sh

echo "[$(date '+%H:%M:%S')] === post (phase A2 + D + analyse) ==="
bash scripts/run_post.sh

echo "[$(date '+%H:%M:%S')] REMAINDER ALL DONE"
