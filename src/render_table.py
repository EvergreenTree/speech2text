"""Render metrics_<lang>.json into a Markdown table that pastes cleanly into the
README. Tag → display name mapping is curated for readability.

Usage:
    python -m src.render_table --metrics outputs/metrics_fr.json --out outputs/table_fr.md
    python -m src.render_table --metrics outputs/metrics_zh.json --out outputs/table_zh.md
"""
from __future__ import annotations

import argparse
import json
import os


# (display_name, trainable_params)
DISPLAY = {
    # ---- fr (FLEURS) ----
    "baseline_tiny":         ("Whisper-tiny **baseline**",                                 "—"),
    "lora_tiny":             ("Whisper-tiny + LoRA (recette zh, LR 1e-4, 1 ep)",          "1,5 M"),
    "full_tiny":             ("Whisper-tiny **full FT**",                                  "39 M"),
    "baseline_small":        ("Whisper-small **baseline**",                                "—"),
    "lora_small":            ("Whisper-small + LoRA (recette zh, LR 1e-4, 1 ep)",         "7,1 M"),
    "lora_small_v2":         ("Whisper-small + LoRA (recette fr, LR 3e-5, 2 ep)",         "7,1 M"),
    "full_small":            ("Whisper-small **full FT**",                                 "244 M"),
    "scratch_small":         ("Whisper-small **from scratch** (random init, 5 ep)",       "244 M"),
    "baseline_medium":       ("Whisper-medium **baseline**",                               "—"),
    "lora_medium":           ("Whisper-medium + LoRA (recette fr)",                        "18,9 M"),
    "baseline_turbo":        ("Whisper-large-v3-turbo **baseline**",                       "—"),
    "lora_turbo":            ("Whisper-large-v3-turbo + LoRA (recette fr)",                "~12 M"),
    "ref_w2v_fr":            ("_ref_ : wav2vec2-CTC-français (zero-shot, paradigm CTC)",   "—"),
    "ref_whisper_fr_distil": ("_ref_ : Whisper-large-v3 distil-fr-dec4 (zero-shot)",       "—"),
    # ---- zh (CV21) ----
    "baseline_tiny_zh":      ("Whisper-tiny **baseline**",                                 "—"),
    "lora_tiny_zh":          ("Whisper-tiny + LoRA (recette zh, LR 1e-4, 1 ep)",          "1,5 M"),
    "full_tiny_zh":          ("Whisper-tiny **full FT**",                                  "39 M"),
}

# zh re-uses tags from the archive; their displays shift slightly.
DISPLAY_ZH_OVERRIDE = {
    "baseline_small":  ("Whisper-small **baseline**",                                "—"),
    "lora_small":      ("Whisper-small + LoRA (recette zh, LR 1e-4, 1 ep)",          "7,1 M"),
    "full_small":      ("Whisper-small **full FT**",                                 "244 M"),
    "baseline_medium": ("Whisper-medium **baseline**",                               "—"),
    "lora_medium":     ("Whisper-medium + LoRA (recette zh, LR 1e-4, 2 ep)",         "18,9 M"),
}

ORDER_FR = [
    "baseline_tiny", "lora_tiny", "full_tiny",
    "baseline_small", "lora_small", "lora_small_v2", "full_small", "scratch_small",
    "baseline_medium", "lora_medium",
    "baseline_turbo", "lora_turbo",
    "ref_w2v_fr", "ref_whisper_fr_distil",
]

ORDER_ZH = [
    "baseline_tiny_zh", "lora_tiny_zh", "full_tiny_zh",
    "baseline_small", "lora_small", "full_small",
    "baseline_medium", "lora_medium",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="/data/speech2text/outputs/metrics_fr.json")
    ap.add_argument("--out", default="/data/speech2text/outputs/table_fr.md")
    args = ap.parse_args()

    with open(args.metrics) as f:
        metrics = json.load(f)
    rows_by_tag = {r["tag"]: r for r in metrics["rows"]}
    language = metrics.get("language", "fr")
    primary = "wer" if language == "fr" else "cer"
    primary_label = "WER" if language == "fr" else "CER"
    abs_key = f"{primary}_abs_delta"
    rel_key = f"{primary}_rel_delta_pct"
    order = ORDER_FR if language == "fr" else ORDER_ZH
    display = DISPLAY if language == "fr" else {**DISPLAY, **DISPLAY_ZH_OVERRIDE}

    lines = []
    if language == "fr":
        lines.append(f"| Run | Trainable | Wall eval (s) | WER | CER | Δ WER abs | Δ WER rel |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append(f"| Run | Trainable | Wall eval (s) | CER | Δ CER abs | Δ CER rel |")
        lines.append("|---|---:|---:|---:|---:|---:|")
    for tag in order:
        r = rows_by_tag.get(tag)
        if r is None:
            continue
        wall = "—" if r.get("wall_seconds", 0) == 0 else f"{r['wall_seconds']:.1f}"
        v_primary = r.get(primary)
        v_cer = r.get("cer")
        primary_str = "—" if v_primary is None else f"**{v_primary:.4f}**"
        cer_str = "—" if v_cer is None else f"{v_cer:.4f}"
        abs_d = r.get(abs_key)
        rel_d = r.get(rel_key)
        abs_str = "—" if abs_d is None else f"{abs_d:+.4f}"
        rel_str = "—" if rel_d is None else f"{rel_d:+.1f} %"
        name, tp = display.get(tag, (tag, "?"))
        if language == "fr":
            lines.append(f"| {name} | {tp} | {wall} | {primary_str} | {cer_str} | {abs_str} | {rel_str} |")
        else:
            # primary == cer, so just one metric column.
            lines.append(f"| {name} | {tp} | {wall} | {primary_str} | {abs_str} | {rel_str} |")
    out = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(out)
    print(out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
