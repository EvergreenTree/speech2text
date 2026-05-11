#!/usr/bin/env bash
set -euo pipefail

PY=/data/venv/bin/python
ROOT=/data/speech2text/Qwen3-ASR/finetuning
OUT=${ROOT}/outputs
DATA=${ROOT}/data
export HF_HOME=/data/speech2text/outputs/cache
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run() {
  local algo="$1" lang="$2" profile="$3" size="$4" extra="${5:-}"
  local lang_short="${lang,,}"; lang_short="${lang_short:0:2}"
  local tag="qwen3_${size}_${algo}_${lang_short}_dev100"
  local out_dir="${OUT}/adapters/qwen3_${size}_${algo}_${lang_short}"
  echo "=== ${tag} ===" ; date -u
  "${PY}" "${ROOT}/qwen3_asr_${algo}.py" \
    --model_path  "Qwen/Qwen3-ASR-0.6B" \
    --train_file  "${DATA}/${profile}/train/train.jsonl" \
    --eval_file   "${DATA}/${profile}/dev/dev.jsonl" \
    --output_dir  "${out_dir}" \
    --tag         "${tag}" \
    --language    "${lang}" \
    --epochs      0.5 --grad_acc 4 --lr 5e-6 \
    --log_steps   10 --eval_steps 200 \
    --eval_out_dir "${OUT}" \
    ${extra}
  echo "=== DONE ${tag} ===" ; date -u
}

run mwer French  fleurs-fr 0p6b "--n_best 4 --mwer_batch_size 4 --generation_strategy sample --temperature 0.9 --top_p 0.95"
run mwer Chinese cv21-zh   0p6b "--n_best 4 --mwer_batch_size 4 --generation_strategy sample --temperature 0.9 --top_p 0.95"
run gspo French  fleurs-fr 0p6b "--group_size 4 --gspo_batch_size 4"
run gspo Chinese cv21-zh   0p6b "--group_size 4 --gspo_batch_size 4"
