#!/usr/bin/env bash
# Run the non-Whisper-family models. Picked up after the Whisper family run
# finishes and after the cache has been moved to /data.
set -e

cd /data/speech2text/asr_bench
LOG=logs/specialists.log
echo "=== specialist run start $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# After move_cache_to_data.sh, /data/speech2text/asr_bench/cache → /data/speech2text/asr_bench/cache.
# Set explicitly here in case the symlink path changes.
export HF_HUB_CACHE=/data/speech2text/asr_bench/cache
export HF_HOME=/data/speech2text/asr_bench/cache

run() {
  local tag=$1; local model=$2; local fam=$3; local lang=$4; local profile=$5; local bsz=$6
  echo ">>> $tag" | tee -a "$LOG"
  /data/venv/bin/python run_eval.py \
    --tag "$tag" --model "$model" \
    --family "$fam" --language "$lang" --lang-profile "$profile" \
    --batch-size "$bsz" --dtype bf16 \
    2>&1 | grep -Ev "max_new_tokens|Both .max_length" | tee -a "$LOG"
}

# --- French: Whisper-arch specialists (cheap) ---
run distilfr_fr     bofenghuang/whisper-large-v3-french-distil-dec4 whisper      fr fleurs_fr 4

# --- French: Voxtral Mini 3B (bf16, fits comfortably) ---
run voxtral_mini_fr mistralai/Voxtral-Mini-3B-2507                  voxtral      fr fleurs_fr 4

# --- French: Voxtral Small 24B in 4-bit (needs bitsandbytes) ---
run voxtral_small4bit_fr mistralai/Voxtral-Small-24B-2507           voxtral_4bit fr fleurs_fr 1

# --- Chinese: Whisper-arch specialists ---
run belle_zh        BELLE-2/Belle-whisper-large-v3-zh               whisper      zh cv21_zh   4
run sensevoice_zh   FunAudioLLM/SenseVoiceSmall                     sensevoice   zh cv21_zh   8

echo "=== specialist run done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
