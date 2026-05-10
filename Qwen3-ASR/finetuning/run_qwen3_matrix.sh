#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/venv/bin/python}"
ROOT="/data/speech2text/Qwen3-ASR/finetuning"
OUT_ROOT="${OUT_ROOT:-/data/speech2text/Qwen3-ASR/finetuning/outputs}"

prepare_profile() {
  local profile="$1"
  "${PYTHON_BIN}" "${ROOT}/prepare_qwen3_asr_data.py" --profile "${profile}"
}

train_and_eval() {
  local model_path="$1"
  local profile="$2"
  local mode="$3"
  local tag="$4"
  local batch_size="$5"
  local grad_acc="$6"
  local lr="$7"
  local epochs="$8"
  local force_language="$9"

  local train_file="${ROOT}/data/${profile}/train/train.jsonl"
  local eval_file="${ROOT}/data/${profile}/dev/dev.jsonl"
  local test_file="${ROOT}/data/${profile}/test/test.jsonl"
  local train_out="${OUT_ROOT}/adapters/${tag}"

  "${PYTHON_BIN}" "${ROOT}/qwen3_asr_sft.py" \
    --model_path "${model_path}" \
    --train_file "${train_file}" \
    --eval_file "${eval_file}" \
    --output_dir "${train_out}" \
    --mode "${mode}" \
    --batch_size "${batch_size}" \
    --grad_acc "${grad_acc}" \
    --lr "${lr}" \
    --epochs "${epochs}" \
    --save_steps 100 \
    --log_steps 10

  if [[ "${mode}" == "lora" ]]; then
    "${PYTHON_BIN}" "${ROOT}/eval_qwen3_asr.py" \
      --model-path "${model_path}" \
      --adapter-path "${train_out}" \
      --jsonl "${test_file}" \
      --tag "${tag}" \
      --force-language "${force_language}"
  else
    "${PYTHON_BIN}" "${ROOT}/eval_qwen3_asr.py" \
      --model-path "${train_out}" \
      --jsonl "${test_file}" \
      --tag "${tag}" \
      --force-language "${force_language}"
  fi
}

prepare_profile fleurs-fr
prepare_profile cv21-zh

# Example invocations; tune batch sizes per GPU memory.
train_and_eval Qwen/Qwen3-ASR-0.6B fleurs-fr lora qwen3_0p6b_lora_fr 2 8 2e-5 1 French
train_and_eval Qwen/Qwen3-ASR-0.6B fleurs-fr full qwen3_0p6b_full_fr 1 8 1e-5 1 French
train_and_eval Qwen/Qwen3-ASR-0.6B cv21-zh lora qwen3_0p6b_lora_zh 2 8 2e-5 1 Chinese
train_and_eval Qwen/Qwen3-ASR-0.6B cv21-zh full qwen3_0p6b_full_zh 1 8 1e-5 1 Chinese
train_and_eval Qwen/Qwen3-ASR-1.7B fleurs-fr lora qwen3_1p7b_lora_fr 1 16 1e-5 1 French
train_and_eval Qwen/Qwen3-ASR-1.7B fleurs-fr full qwen3_1p7b_full_fr 1 16 5e-6 1 French
train_and_eval Qwen/Qwen3-ASR-1.7B cv21-zh lora qwen3_1p7b_lora_zh 1 16 1e-5 1 Chinese
train_and_eval Qwen/Qwen3-ASR-1.7B cv21-zh full qwen3_1p7b_full_zh 1 16 5e-6 1 Chinese
