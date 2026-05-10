#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/speech2text/granite_speech/finetuning"
QWEN_ROOT="/data/speech2text/Qwen3-ASR/finetuning"
OUT_ROOT="${OUT_ROOT:-/data/speech2text/granite_speech/finetuning/outputs}"
PY="/data/venv/bin/python"
MODEL="ibm-granite/granite-speech-4.1-2b"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/adapters" "${OUT_ROOT}/preds" "${OUT_ROOT}/logs"
cd "$ROOT"

eval_only() {
  local jsonl="$1"
  local tag="$2"
  local batch="$3"
  ${PY} "$ROOT/eval_granite_speech.py" \
    --model-path "$MODEL" \
    --jsonl "$jsonl" \
    --tag "$tag" \
    --out-dir "${OUT_ROOT}/preds" \
    --batch-size "$batch" \
    2>&1 | tee "${OUT_ROOT}/logs/${tag}.log"
}

train_and_eval() {
  local profile="$1"
  local mode="$2"
  local tag="$3"
  local batch="$4"
  local grad_acc="$5"
  local lr="$6"
  local epochs="$7"
  local eval_batch="$8"
  local train_file="${QWEN_ROOT}/data/${profile}/train/train.jsonl"
  local dev_file="${QWEN_ROOT}/data/${profile}/dev/dev.jsonl"
  local eval_jsonl="${QWEN_ROOT}/eval_slices/$(basename "$9")"
  local out_dir="${OUT_ROOT}/adapters/${tag}"

  ${PY} "$ROOT/granite_speech_sft.py" \
    --model_path "$MODEL" \
    --train_file "$train_file" \
    --eval_file "$dev_file" \
    --output_dir "$out_dir" \
    --mode "$mode" \
    --batch_size "$batch" \
    --grad_acc "$grad_acc" \
    --lr "$lr" \
    --epochs "$epochs" \
    --gradient_checkpointing 1 \
    2>&1 | tee "${OUT_ROOT}/logs/train_${tag}.log"

  if [[ "$mode" == "lora" ]]; then
    ${PY} "$ROOT/eval_granite_speech.py" \
      --model-path "$MODEL" \
      --adapter-path "$out_dir" \
      --jsonl "$eval_jsonl" \
      --tag "${tag}_dev100" \
      --out-dir "${OUT_ROOT}/preds" \
      --batch-size "$eval_batch" \
      2>&1 | tee "${OUT_ROOT}/logs/eval_${tag}.log"
  else
    ${PY} "$ROOT/eval_granite_speech.py" \
      --model-path "$out_dir" \
      --jsonl "$eval_jsonl" \
      --tag "${tag}_dev100" \
      --out-dir "${OUT_ROOT}/preds" \
      --batch-size "$eval_batch" \
      2>&1 | tee "${OUT_ROOT}/logs/eval_${tag}.log"
  fi
}

eval_only "${QWEN_ROOT}/eval_slices/fleurs_fr_dev100.jsonl" "granite_speech_2b_base_fr_dev100" 8
eval_only "${QWEN_ROOT}/eval_slices/cv21_zh_dev100.jsonl" "granite_speech_2b_base_zh_dev100" 8

train_and_eval "fleurs-fr" "lora" "granite_speech_2b_lora_fr" 4 4 1e-5 1 8 "fleurs_fr_dev100.jsonl"
train_and_eval "cv21-zh" "lora" "granite_speech_2b_lora_zh" 4 4 1e-5 1 8 "cv21_zh_dev100.jsonl"
train_and_eval "fleurs-fr" "full" "granite_speech_2b_full_fr" 1 16 5e-6 1 8 "fleurs_fr_dev100.jsonl"
train_and_eval "cv21-zh" "full" "granite_speech_2b_full_zh" 1 16 5e-6 1 8 "cv21_zh_dev100.jsonl"
