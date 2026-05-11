#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_rl_matrix.sh  —  MWER + GSPO training matrix for Qwen3-ASR
#
# Runs 8 combinations: {0.6B, 1.7B} × {MWER, GSPO} × {FR, ZH}
# Each run trains for 1 epoch and evaluates on the first 100 dev examples
# (same slice as the existing SFT results in outputs/*.json).
#
# Output JSONs land in:
#   outputs/qwen3_{size}_{algo}_{lang}_dev100.json
#
# Usage:
#   bash run_rl_matrix.sh               # full matrix
#   bash run_rl_matrix.sh 0p6b_only     # 0.6B only (faster)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/venv/bin/python}"
ROOT="/data/speech2text/Qwen3-ASR/finetuning"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs}"
DATA_ROOT="${ROOT}/data"
MODE="${1:-full}"   # "full" | "0p6b_only"

# ── Helper ────────────────────────────────────────────────────────────────────
run_mwer() {
  local model_path="$1"
  local profile="$2"         # fleurs-fr | cv21-zh
  local language="$3"        # French | Chinese
  local size_tag="$4"        # 0p6b | 1p7b
  local n_best="$5"
  local lr="$6"
  local mwer_batch_size="$7"

  local lang_short="${language,,}" # lower-case
  lang_short="${lang_short:0:2}"   # fr | zh
  local tag="qwen3_${size_tag}_mwer_${lang_short}_dev100"
  local adapter_dir="${OUT_ROOT}/adapters/qwen3_${size_tag}_mwer_${lang_short}"

  echo "━━━ MWER  ${model_path}  ${profile} ━━━"
  "${PYTHON_BIN}" "${ROOT}/qwen3_asr_mwer.py" \
    --model_path  "${model_path}" \
    --train_file  "${DATA_ROOT}/${profile}/train/train.jsonl" \
    --eval_file   "${DATA_ROOT}/${profile}/dev/dev.jsonl" \
    --output_dir  "${adapter_dir}" \
    --tag         "${tag}" \
    --language    "${language}" \
    --n_best      "${n_best}" \
    --mwer_batch_size "${mwer_batch_size}" \
    --generation_strategy sample \
    --temperature 0.9 \
    --top_p       0.95 \
    --lambda_ce   0.01 \
    --lr          "${lr}" \
    --epochs      1 \
    --grad_acc    4 \
    --log_steps   50 \
    --eval_out_dir "${OUT_ROOT}"
}

run_gspo() {
  local model_path="$1"
  local profile="$2"
  local language="$3"
  local size_tag="$4"
  local group_size="$5"
  local lr="$6"

  local lang_short="${language,,}"
  lang_short="${lang_short:0:2}"
  local tag="qwen3_${size_tag}_gspo_${lang_short}_dev100"
  local adapter_dir="${OUT_ROOT}/adapters/qwen3_${size_tag}_gspo_${lang_short}"

  echo "━━━ GSPO  ${model_path}  ${profile} ━━━"
  "${PYTHON_BIN}" "${ROOT}/qwen3_asr_gspo.py" \
    --model_path  "${model_path}" \
    --train_file  "${DATA_ROOT}/${profile}/train/train.jsonl" \
    --eval_file   "${DATA_ROOT}/${profile}/dev/dev.jsonl" \
    --output_dir  "${adapter_dir}" \
    --tag         "${tag}" \
    --language    "${language}" \
    --group_size  "${group_size}" \
    --gspo_batch_size "${group_size}" \
    --format_alpha 0.1 \
    --lr          "${lr}" \
    --epochs      1 \
    --grad_acc    4 \
    --log_steps   50 \
    --eval_out_dir "${OUT_ROOT}"
}

# ── 0.6B runs ─────────────────────────────────────────────────────────────────
run_mwer "Qwen/Qwen3-ASR-0.6B" "fleurs-fr" "French"  "0p6b" 4 "5e-6" 2
run_mwer "Qwen/Qwen3-ASR-0.6B" "cv21-zh"   "Chinese" "0p6b" 4 "5e-6" 2
run_gspo "Qwen/Qwen3-ASR-0.6B" "fleurs-fr" "French"  "0p6b" 4 "5e-6"
run_gspo "Qwen/Qwen3-ASR-0.6B" "cv21-zh"   "Chinese" "0p6b" 4 "5e-6"

if [[ "${MODE}" != "0p6b_only" ]]; then
  # ── 1.7B runs (smaller groups due to memory) ────────────────────────────────
  run_mwer "Qwen/Qwen3-ASR-1.7B" "fleurs-fr" "French"  "1p7b" 2 "2e-6" 2
  run_mwer "Qwen/Qwen3-ASR-1.7B" "cv21-zh"   "Chinese" "1p7b" 2 "2e-6" 2
  run_gspo "Qwen/Qwen3-ASR-1.7B" "fleurs-fr" "French"  "1p7b" 2 "2e-6"
  run_gspo "Qwen/Qwen3-ASR-1.7B" "cv21-zh"   "Chinese" "1p7b" 2 "2e-6"
fi

echo "━━━ RL matrix complete ━━━"
echo "Results in ${OUT_ROOT}/"
