#!/usr/bin/env bash
# Phase tiny: baseline / LoRA / full FT for Whisper-tiny on both fr (FLEURS) and
# zh (CV21). Adds 3 + 3 = 6 rows to the cross-language tables.
#
# fr re-uses outputs/cache/processed/openai__whisper-small/processed (tokenizer
#   and features are bit-identical between tiny / small / medium).
# zh re-prepares the dataset (was deleted to free disk after the original run);
#   we encode under outputs/cache/processed/cv21-zh/.
set -euo pipefail
VENV=/data/venv
PY="$VENV/bin/python"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
ROOT=/data/speech2text
cd "$ROOT"

TINY=openai/whisper-tiny

# --- fr: reuse the small-encoded splits ----
PROC_FR="outputs/cache/processed/openai__whisper-small/processed"
RAW_FR="outputs/cache/processed/openai__whisper-small/raw"

echo "[$(date '+%H:%M:%S')] === baseline_tiny (fr) ==="
$PY -m src.eval --model "$TINY" --raw-dir "$RAW_FR" --tag baseline_tiny \
    --language fr --batch-size 32 \
    2>&1 | tee outputs/logs/eval_baseline_tiny.log

echo "[$(date '+%H:%M:%S')] === train lora_tiny (fr) ==="
$PY -m src.train --model "$TINY" --mode lora \
    --processed-dir "$PROC_FR" --out outputs/adapters/lora_tiny \
    --batch 32 --grad-accum 1 --epochs 1 --lr 5e-5 --language fr \
    --lora-rank 32 --lora-alpha 64 \
    2>&1 | tee outputs/logs/train_lora_tiny.log

echo "[$(date '+%H:%M:%S')] === eval lora_tiny (fr) ==="
$PY -m src.eval --model "$TINY" --adapter outputs/adapters/lora_tiny \
    --raw-dir "$RAW_FR" --tag lora_tiny --language fr --batch-size 32 \
    2>&1 | tee outputs/logs/eval_lora_tiny.log

echo "[$(date '+%H:%M:%S')] === train full_tiny (fr) ==="
$PY -m src.train --model "$TINY" --mode full \
    --processed-dir "$PROC_FR" --out outputs/adapters/full_tiny \
    --batch 32 --grad-accum 1 --epochs 1 --lr 1e-5 --language fr \
    2>&1 | tee outputs/logs/train_full_tiny.log

echo "[$(date '+%H:%M:%S')] === eval full_tiny (fr) ==="
$PY -m src.eval --model outputs/adapters/full_tiny --raw-dir "$RAW_FR" \
    --tag full_tiny --language fr --batch-size 32 \
    2>&1 | tee outputs/logs/eval_full_tiny.log

# --- zh: encode CV21 once, then run the three runs ----
echo "[$(date '+%H:%M:%S')] === encode zh CV21 ==="
$PY -m src.data --model-id "$TINY" --profile cv21-zh \
    --n-train 4000 --n-dev 300 --n-test 500 --num-proc 4 \
    --language zh \
    2>&1 | tee outputs/logs/dataprep_zh.log

PROC_ZH="outputs/cache/processed/cv21-zh/openai__whisper-tiny/processed"
RAW_ZH="outputs/cache/processed/cv21-zh/openai__whisper-tiny/raw"

echo "[$(date '+%H:%M:%S')] === baseline_tiny_zh ==="
$PY -m src.eval --model "$TINY" --raw-dir "$RAW_ZH" --tag baseline_tiny_zh \
    --language zh --batch-size 32 \
    2>&1 | tee outputs/logs/eval_baseline_tiny_zh.log

echo "[$(date '+%H:%M:%S')] === train lora_tiny_zh ==="
# Use the original zh recipe (LR 1e-4, 1 epoch) — it worked on small / medium.
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

echo "[$(date '+%H:%M:%S')] PHASE TINY DONE"
