#!/usr/bin/env bash
# Final tail of the queue: re-run Canary with auto-detect prompt + Voxtral-Small.
set -e
cd /data/speech2text/asr_bench
LOG=logs/remainder.log
echo "=== final remainder run start $(date -u +%FT%TZ) ===" | tee -a "$LOG"

export HF_HUB_CACHE=/data/speech2text/asr_bench/cache
export HF_HOME=/data/speech2text/asr_bench/cache

# 1. Canary-Qwen 2.5B (re-run with fixed auto-detect prompt) — fast.
/data/nemo_venv/bin/python run_nemo.py \
  --tag canary_qwen_fr --model nvidia/canary-qwen-2.5b \
  --family canary --language fr --lang-profile fleurs_fr \
  --batch-size 4 --force \
  2>&1 | tee -a "$LOG" || true

# 2. Voxtral Small 24B 4-bit — slowest, last.
/data/venv/bin/python run_eval.py \
  --tag voxtral_small4bit_fr --model mistralai/Voxtral-Small-24B-2507 \
  --family voxtral_4bit --language fr --lang-profile fleurs_fr \
  --batch-size 1 --dtype bf16 --force \
  2>&1 | grep -Ev "max_new_tokens|Both .max_length|^Loading weights|^Fetching" | tee -a "$LOG" || true

echo "=== final remainder run done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
