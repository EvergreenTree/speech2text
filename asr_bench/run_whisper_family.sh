#!/usr/bin/env bash
# Run the OpenAI Whisper family (7 sizes) on both fleurs_fr and cv21_zh.
# Smaller models batch-16; larger ones drop the batch to keep VRAM in budget.
set -e

cd /data/speech2text/asr_bench
LOG=logs/whisper_family.log
echo "=== whisper family run start $(date -u +%FT%TZ) ===" | tee -a "$LOG"

run() {
  local tag=$1
  local model=$2
  local lang=$3
  local profile=$4
  local bsz=$5
  echo ">>> $tag" | tee -a "$LOG"
  /data/venv/bin/python run_eval.py \
    --tag "$tag" --model "$model" \
    --family whisper --language "$lang" --lang-profile "$profile" \
    --batch-size "$bsz" --dtype bf16 \
    2>&1 | grep -Ev "max_new_tokens|Both .max_length" | tee -a "$LOG"
}

# --- French ---
run tiny_fr     openai/whisper-tiny          fr fleurs_fr 16
run base_fr     openai/whisper-base          fr fleurs_fr 16
run small_fr    openai/whisper-small         fr fleurs_fr 16
run medium_fr   openai/whisper-medium        fr fleurs_fr 8
run largev2_fr  openai/whisper-large-v2      fr fleurs_fr 4
run largev3_fr  openai/whisper-large-v3      fr fleurs_fr 4
run turbo_fr    openai/whisper-large-v3-turbo fr fleurs_fr 8

# --- Chinese ---
run tiny_zh     openai/whisper-tiny          zh cv21_zh 16
run base_zh     openai/whisper-base          zh cv21_zh 16
run small_zh    openai/whisper-small         zh cv21_zh 16
run medium_zh   openai/whisper-medium        zh cv21_zh 8
run largev2_zh  openai/whisper-large-v2      zh cv21_zh 4
run largev3_zh  openai/whisper-large-v3      zh cv21_zh 4
run turbo_zh    openai/whisper-large-v3-turbo zh cv21_zh 8

echo "=== whisper family run done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
