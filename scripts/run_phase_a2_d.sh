#!/usr/bin/env bash
# Phases A2 (LoRA-small avec LR conservatif) + D (référence wav2vec2-fr off-the-shelf).
# À lancer une fois Phase C terminé.
set -euo pipefail
ROOT=/data/speech2text
cd "$ROOT"
echo "[$(date '+%H:%M:%S')] starting phase A2"
bash scripts/run_phase_a2.sh
echo "[$(date '+%H:%M:%S')] starting phase D"
bash scripts/run_phase_d.sh
echo "[$(date '+%H:%M:%S')] running analysis"
/data/venv/bin/python -m src.analyze --preds-dir outputs/preds --out-dir outputs
echo "[$(date '+%H:%M:%S')] PHASE A2+D ALL DONE"
