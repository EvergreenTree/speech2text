#!/usr/bin/env bash
# Sequential RL training: run each job, wait for its output JSON, then start next.
# Designed to survive SSH session loss (run with setsid).
set -uo pipefail

PY=/data/venv/bin/python
ROOT=/data/speech2text/Qwen3-ASR/finetuning
OUT=${ROOT}/outputs
DATA=${ROOT}/data
LOG_DIR=${OUT}/logs
ADAPTERS=${OUT}/adapters
export HF_HOME=/data/speech2text/outputs/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${LOG_DIR}" "${ADAPTERS}"

run_job() {
  local algo="$1" lang="$2" profile="$3" size="$4" epochs="$5" extra="${6:-}"
  local lang_short="${lang,,}"; lang_short="${lang_short:0:2}"
  local tag="qwen3_${size}_${algo}_${lang_short}_dev100"
  local out_json="${OUT}/${tag}.json"
  local adapter_dir="${ADAPTERS}/qwen3_${size}_${algo}_${lang_short}"
  local log="${LOG_DIR}/${tag}_$(date -u +%Y%m%d_%H%M%S).log"

  if [[ -f "${out_json}" ]]; then
    echo "[skip] ${tag} already done → ${out_json}"
    return 0
  fi

  echo "[start] ${tag}  $(date -u)"
  "${PY}" "${ROOT}/qwen3_asr_${algo}.py" \
    --model_path  "Qwen/Qwen3-ASR-0.6B" \
    --train_file  "${DATA}/${profile}/train/train.jsonl" \
    --eval_file   "${DATA}/${profile}/dev/dev.jsonl" \
    --output_dir  "${adapter_dir}" \
    --tag         "${tag}" \
    --language    "${lang}" \
    --epochs      "${epochs}" \
    --grad_acc    4 --lr 5e-6 \
    --log_steps   25 --eval_steps 0 \
    --eval_out_dir "${OUT}" \
    ${extra} \
    > "${log}" 2>&1
  local ec=$?
  echo "[done] ${tag} exit=${ec}  $(date -u)"
  if [[ -f "${out_json}" ]]; then
    echo "[ok] result saved to ${out_json}"
    cat "${out_json}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  WER={d.get(\"wer\",\"n/a\")}  CER={d.get(\"cer\",\"n/a\")}  n={d.get(\"n\",\"?\")}')";
  else
    echo "[warn] no output JSON at ${out_json}"
  fi
}

# Skip GSPO-FR — already running (PID 1304055)
# Queue: MWER-FR, MWER-ZH, GSPO-ZH
# Wait for GSPO-FR to finish first (it holds 11GB GPU)
echo "Waiting for GSPO-FR (PID 1304055) to finish..."
while kill -0 1304055 2>/dev/null; do sleep 30; done
echo "GSPO-FR done. Starting MWER-FR..."

run_job mwer French  fleurs-fr 0p6b 0.25 "--n_best 4 --mwer_batch_size 1"
run_job mwer Chinese cv21-zh   0p6b 0.25 "--n_best 4 --mwer_batch_size 1"
run_job gspo Chinese cv21-zh   0p6b 0.25 "--group_size 4"

echo "All jobs complete."
