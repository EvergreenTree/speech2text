#!/usr/bin/env bash
# Post-main pipeline : Phase A2 (lora_small avec LR conservatif) + Phase D
# (référence externe wav2vec2-fr) + analyse finale.
set -euo pipefail
ROOT=/data/speech2text
cd "$ROOT"

echo "[$(date '+%H:%M:%S')] phase A2 (lora_small_v2 with LR 3e-5, 2 ep)"
bash scripts/run_phase_a2.sh

echo "[$(date '+%H:%M:%S')] phase D (off-the-shelf wav2vec2 zero-shot)"
bash scripts/run_phase_d.sh

echo "[$(date '+%H:%M:%S')] running final analysis"
/data/venv/bin/python -m src.analyze --preds-dir outputs/preds --out-dir outputs

echo "[$(date '+%H:%M:%S')] rendering markdown table"
/data/venv/bin/python -m src.render_table --metrics outputs/metrics.json --out outputs/table.md

echo "[$(date '+%H:%M:%S')] POST-PIPELINE ALL DONE"
cat outputs/metrics.json | /data/venv/bin/python -m json.tool
echo "---"
cat outputs/table.md
