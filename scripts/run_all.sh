#!/usr/bin/env bash
# End-to-end pipeline (FLEURS fr_fr): data prep -> baseline / LoRA / full / scratch on small
# -> baseline / LoRA on medium -> baseline / LoRA on large-v3-turbo -> analysis.
# Run from /data/speech2text/.

set -euo pipefail

VENV=/data/venv
PY="$VENV/bin/python"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

ROOT=/data/speech2text
cd "$ROOT"

SMALL=openai/whisper-small
MEDIUM=openai/whisper-medium
TURBO=openai/whisper-large-v3-turbo

# ---- Phase 1: data prep (small) ----
$PY -m src.data --model-id "$SMALL" --n-test 500 --num-proc 4

PROC_S="outputs/cache/processed/openai__whisper-small/processed"
RAW_S="outputs/cache/processed/openai__whisper-small/raw"

# ---- Phase A: bootstrap on whisper-small ----
$PY -m src.eval --model "$SMALL" --raw-dir "$RAW_S" --tag baseline_small --batch-size 16

$PY -m src.train --model "$SMALL" --mode lora \
    --processed-dir "$PROC_S" --out outputs/adapters/lora_small \
    --batch 16 --grad-accum 1 --epochs 1 --lr 1e-4 \
    --lora-rank 32 --lora-alpha 64

$PY -m src.eval --model "$SMALL" --adapter outputs/adapters/lora_small \
    --raw-dir "$RAW_S" --tag lora_small --batch-size 16

$PY -m src.train --model "$SMALL" --mode full \
    --processed-dir "$PROC_S" --out outputs/adapters/full_small \
    --batch 8 --grad-accum 2 --epochs 1 --lr 1e-5 --gradient-checkpointing

$PY -m src.eval --model outputs/adapters/full_small --raw-dir "$RAW_S" \
    --tag full_small --batch-size 16

# Random-init from scratch (same architecture as small). Higher LR, more
# epochs, more warmup — pretraining is the only thing missing, so we give it
# a fair training budget at the cost of compute.
$PY -m src.train --model "$SMALL" --mode scratch \
    --processed-dir "$PROC_S" --out outputs/adapters/scratch_small \
    --batch 16 --grad-accum 1 --epochs 10 --lr 5e-4 --warmup-ratio 0.05 \
    --gradient-checkpointing

$PY -m src.eval --model outputs/adapters/scratch_small --raw-dir "$RAW_S" \
    --tag scratch_small --batch-size 16

# ---- Phase B: scale to whisper-medium ----
$PY -m src.data --model-id "$MEDIUM" --n-test 500 --num-proc 4

PROC_M="outputs/cache/processed/openai__whisper-medium/processed"
RAW_M="outputs/cache/processed/openai__whisper-medium/raw"

$PY -m src.eval --model "$MEDIUM" --raw-dir "$RAW_M" --tag baseline_medium --batch-size 8

$PY -m src.train --model "$MEDIUM" --mode lora \
    --processed-dir "$PROC_M" --out outputs/adapters/lora_medium \
    --batch 8 --grad-accum 2 --epochs 2 --lr 1e-4 \
    --lora-rank 32 --lora-alpha 64 --gradient-checkpointing

$PY -m src.eval --model "$MEDIUM" --adapter outputs/adapters/lora_medium \
    --raw-dir "$RAW_M" --tag lora_medium --batch-size 8

# ---- Phase C: chase SoTA under constraint — Whisper-large-v3-turbo + LoRA ----
$PY -m src.data --model-id "$TURBO" --n-test 500 --num-proc 4

PROC_T="outputs/cache/processed/openai__whisper-large-v3-turbo/processed"
RAW_T="outputs/cache/processed/openai__whisper-large-v3-turbo/raw"

$PY -m src.eval --model "$TURBO" --raw-dir "$RAW_T" --tag baseline_turbo --batch-size 4

$PY -m src.train --model "$TURBO" --mode lora \
    --processed-dir "$PROC_T" --out outputs/adapters/lora_turbo \
    --batch 4 --grad-accum 4 --epochs 2 --lr 1e-4 \
    --lora-rank 32 --lora-alpha 64 --gradient-checkpointing

$PY -m src.eval --model "$TURBO" --adapter outputs/adapters/lora_turbo \
    --raw-dir "$RAW_T" --tag lora_turbo --batch-size 4

# ---- Analysis ----
$PY -m src.analyze --preds-dir outputs/preds --out-dir outputs

echo "DONE. Final metrics:"
cat outputs/metrics.json | $PY -m json.tool
