#!/usr/bin/env bash
# Show one-line summary of every pred in outputs/preds/.
set -euo pipefail
cd /data/speech2text
/data/venv/bin/python - <<'PY'
import json, os
preds_dir = "/data/speech2text/outputs/preds"
rows = []
for f in sorted(os.listdir(preds_dir)):
    if not f.endswith(".json"):
        continue
    with open(os.path.join(preds_dir, f)) as fh:
        d = json.load(fh)
    rows.append((f[:-5], d.get("wer", 0), d.get("cer", 0), d.get("wall_seconds", 0), d.get("n", 0)))
rows.sort(key=lambda r: (r[2] if r[1] is None else r[1]))
print(f"{'tag':<26} {'WER':>7} {'CER':>7} {'sec':>6} {'n':>4}")
print("-"*56)
for tag, wer, cer, sec, n in rows:
    w = "  -   " if wer is None else f"{wer:7.4f}"
    c = "  -   " if cer is None else f"{cer:7.4f}"
    print(f"{tag:<26} {w} {c} {sec:6.1f} {n:4}")
PY
