#!/usr/bin/env bash
# Run all three phases sequentially, then analyze. Assumes Phase 1 data prep
# (small) is already done. Logs everything under outputs/logs/.
set -euo pipefail
ROOT=/data/speech2text
cd "$ROOT"
echo "[$(date '+%H:%M:%S')] starting phase A"
bash scripts/run_phase_a.sh
echo "[$(date '+%H:%M:%S')] starting phase B"
bash scripts/run_phase_b.sh
echo "[$(date '+%H:%M:%S')] starting phase C"
bash scripts/run_phase_c.sh
echo "[$(date '+%H:%M:%S')] running analysis"
/data/venv/bin/python -m src.analyze --preds-dir outputs/preds --out-dir outputs
echo "[$(date '+%H:%M:%S')] ALL DONE"
cat outputs/metrics.json | /data/venv/bin/python -m json.tool
