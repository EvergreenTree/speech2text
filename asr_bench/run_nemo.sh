#!/usr/bin/env bash
# Run NVIDIA NeMo ASR models in /data/nemo_venv. Picks up after specialists.
set -e

cd /data/speech2text/asr_bench
LOG=logs/nemo.log
echo "=== nemo run start $(date -u +%FT%TZ) ===" | tee -a "$LOG"

export HF_HUB_CACHE=/data/speech2text/asr_bench/cache
export HF_HOME=/data/speech2text/asr_bench/cache

run() {
  local tag=$1; local model=$2; local fam=$3; local lang=$4; local profile=$5; local bsz=$6
  echo ">>> $tag" | tee -a "$LOG"
  /data/nemo_venv/bin/python run_nemo.py \
    --tag "$tag" --model "$model" \
    --family "$fam" --language "$lang" --lang-profile "$profile" \
    --batch-size "$bsz" \
    2>&1 | tee -a "$LOG"
}

# Parakeet-TDT 0.6B v3 (multilingual European, 25 langs incl. fr)
run parakeet_tdt_fr  nvidia/parakeet-tdt-0.6b-v3  parakeet fr fleurs_fr 8

# Canary-Qwen 2.5B (en/fr/de/es)
run canary_qwen_fr   nvidia/canary-qwen-2.5b      canary   fr fleurs_fr 4

echo "=== nemo run done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
