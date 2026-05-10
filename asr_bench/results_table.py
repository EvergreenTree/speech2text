"""Render a Markdown table summarizing all preds/*.json results."""
from __future__ import annotations

import glob
import json
import os

PREDS_DIR = "/data/speech2text/asr_bench/preds"


def load_all() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(PREDS_DIR, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "tag": r["tag"],
            "model_id": r["model_id"],
            "family": r["family"],
            "language": r["language"],
            "n": r["n"],
            "n_params": r["n_params"],
            "wer": r["wer"],
            "cer": r["cer"],
            "wall_seconds": r["wall_seconds"],
        })
    return rows


def fmt_pct(x):
    return "—" if x is None else f"{x * 100:.2f}%"


def fmt_params(n):
    if n is None or n <= 0:
        return "?"
    if n >= 1e9:
        return f"{n / 1e9:.2f} B"
    return f"{n / 1e6:.0f} M"


CAVEAT = {
    "canary_qwen_fr": "**English-only model** — French eval is a language-mismatch baseline (model card explicitly states it does not support fr/de/es transcription, only en). Useful as a starting point for measuring future fine-tuning gain.",
    "voxtral_small4bit_fr": "Run in 4-bit (NF4) via bitsandbytes due to L4 23 GB VRAM cap. Quantization typically adds ~0.2-0.5 pt WER vs fp16.",
}


def render_table(rows, lang):
    rows = [r for r in rows if r["language"] == lang]
    rows.sort(key=lambda r: (r["n_params"] or 0, r["tag"]))
    lines = []
    lines.append(f"### {'fr-FR FLEURS' if lang == 'fr' else 'zh-CN Common Voice 21'} ({rows[0]['n'] if rows else 0} test utterances)")
    lines.append("")
    if lang == "fr":
        lines.append("| Tag | Model | Family | Params | WER | CER | Wall (s) |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for r in rows:
            lines.append(f"| `{r['tag']}` | `{r['model_id']}` | {r['family']} | "
                         f"{fmt_params(r['n_params'])} | "
                         f"{fmt_pct(r['wer'])} | {fmt_pct(r['cer'])} | "
                         f"{r['wall_seconds']:.0f} |")
    else:
        lines.append("| Tag | Model | Family | Params | CER | Wall (s) |")
        lines.append("|---|---|---|---:|---:|---:|")
        for r in rows:
            lines.append(f"| `{r['tag']}` | `{r['model_id']}` | {r['family']} | "
                         f"{fmt_params(r['n_params'])} | "
                         f"{fmt_pct(r['cer'])} | "
                         f"{r['wall_seconds']:.0f} |")
    # Footnotes for any rows that have caveats.
    notes = [(r["tag"], CAVEAT[r["tag"]]) for r in rows if r["tag"] in CAVEAT]
    if notes:
        lines.append("")
        lines.append("**Notes:**")
        for tag, note in notes:
            lines.append(f"- `{tag}`: {note}")
    return "\n".join(lines)


def main():
    rows = load_all()
    out = []
    out.append("# ASR benchmark — fr-FR / zh-CN")
    out.append("")
    out.append(render_table(rows, "fr"))
    out.append("")
    out.append(render_table(rows, "zh"))
    out.append("")
    text = "\n".join(out)
    print(text)
    out_path = "/data/speech2text/asr_bench/results.md"
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
