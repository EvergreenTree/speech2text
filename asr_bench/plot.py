"""WER/CER vs Model Size — fr-FR FLEURS and zh-CN CV21.

Combines:
  - speech2text/asr_bench/preds/*.json        : baseline runs from the bench
  - speech2text/outputs/metrics*.json         : Whisper fine-tunes
  - speech2text/Qwen3-ASR/finetuning/outputs  : Qwen baselines + fine-tunes

Plots every metric summarized in the README tables, except intentionally
suppressed outliers that would collapse the useful y-scale.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

PREDS_DIR = "/data/speech2text/asr_bench/preds"
FIG_DIR = "/data/speech2text/asr_bench/figures"
FR_METRICS = "/data/speech2text/outputs/metrics_fr.json"
ZH_METRICS = "/data/speech2text/outputs/metrics_zh.json"
QWEN_PREDS_DIR = "/data/speech2text/Qwen3-ASR/finetuning/outputs/preds"
OUTPUT_PREDS_DIR = "/data/speech2text/outputs/preds"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Display configuration ----------

DISPLAY = {
    "tiny":              "Whisper Tiny, OpenAI",
    "base":              "Whisper Base, OpenAI",
    "small":             "Whisper Small, OpenAI",
    "medium":            "Whisper Medium, OpenAI",
    "largev2":           "Whisper Large v2, OpenAI",
    "largev3":           "Whisper Large v3, OpenAI",
    "turbo":             "Whisper Large v3 Turbo, OpenAI",
    "distilfr":          "distil-large-v3-fr, bofenghuang",
    "voxtral_mini":      "Voxtral Mini 3B, Mistral",
    "voxtral_small4bit": "Voxtral Small 24B (4-bit), Mistral",
    "belle":             "Belle-whisper-large-v3-zh, BELLE",
    "sensevoice":        "SenseVoice Small, Alibaba",
    "parakeet_tdt":      "Parakeet-TDT 0.6B v3, NVIDIA",
    "canary_qwen":       "Canary-Qwen 2.5B, NVIDIA (English-only)",
    "qwen3_0p6b":        "Qwen3-ASR 0.6B, Alibaba",
    "qwen3_1p7b":        "Qwen3-ASR 1.7B, Alibaba",
    "ref_w2v_fr":        "wav2vec2-CTC FR, bofenghuang",
}

IN_SCOPE_COLOR = "#5aa9e6"
OUT_SCOPE_COLOR = "#8f98a3"
FULL_FT_COLOR = "#8b1e3f"
LORA_COLOR = "#f28e2b"
ARROW_COLOR = "#6f7782"

PARAMS_OVERRIDE_B = {
    "voxtral_small4bit": 23.6,
    "ref_w2v_fr": 0.315,
}

FT_SCOPE_TAGS = {
    "tiny",
    "small",
    "medium",
    "turbo",
    "qwen3_0p6b",
    "qwen3_1p7b",
}

Y_LIMITS = {
    "fr": 50.0,
    "zh": 64.0,
}

# Hand-tuned label offsets (dx pts, dy pts, ha) per language.
OFFSETS_FR = {
    "tiny":              ( 14,   0, "left"),
    "base":              ( 14,   0, "left"),
    "small":             ( 14,   0, "left"),
    "parakeet_tdt":      (-14, -34, "right"),
    "medium":            ( 18,  22, "left"),
    "distilfr":          ( 18,  -8, "left"),
    "turbo":             (-18,  -8, "right"),
    "largev2":           ( 18,  22, "left"),
    "largev3":           ( 18, -22, "left"),
    "voxtral_mini":      ( 14, -22, "left"),
    "voxtral_small4bit": (-12,  24, "right"),
    "canary_qwen":       ( 18,   0, "left"),
    "qwen3_0p6b":        ( 18,  18, "left"),
    "qwen3_1p7b":        (-18, -10, "right"),
    "ref_w2v_fr":        ( 18,  18, "left"),
}
OFFSETS_ZH = {
    "tiny":              ( 14,   0, "left"),
    "base":              ( 14,   0, "left"),
    "sensevoice":        ( 14,   0, "left"),
    "small":             ( 14,   0, "left"),
    "medium":            (-14,   0, "right"),
    "turbo":             ( 14, -22, "left"),
    "largev2":           ( 14,   0, "left"),
    "largev3":           ( 18,  22, "left"),
    "belle":             ( 18, -22, "left"),
    "qwen3_0p6b":        ( 18,  10, "left"),
    "qwen3_1p7b":        (-18, -12, "right"),
}


# ---------- Data loading ----------


def collect_baselines(language: str) -> list[dict]:
    suffix = f"_{language}.json"
    rows = []
    for path in sorted(glob.glob(os.path.join(PREDS_DIR, f"*{suffix}"))):
        with open(path) as f:
            r = json.load(f)
        base = os.path.basename(path)[: -len(suffix)]
        params_b = (PARAMS_OVERRIDE_B[base] if base in PARAMS_OVERRIDE_B
                    else (r["n_params"] / 1e9 if r["n_params"] > 0 else None))
        rows.append({
            "tag_root": base,
            "name": DISPLAY.get(base, base),
            "params_b": params_b,
            "wer": r["wer"],
            "cer": r["cer"],
        })
    return rows


def load_extra_readme_baselines(language: str) -> list[dict]:
    if language != "fr":
        return []
    path = os.path.join(OUTPUT_PREDS_DIR, "ref_w2v_fr.json")
    with open(path) as f:
        r = json.load(f)
    return [{
        "tag_root": "ref_w2v_fr",
        "name": DISPLAY["ref_w2v_fr"],
        "params_b": PARAMS_OVERRIDE_B["ref_w2v_fr"],
        "wer": r.get("wer"),
        "cer": r.get("cer"),
    }]


QWEN_BASELINES = {
    "fr": [
        ("qwen3_0p6b_base_fr_dev100.json", "qwen3_0p6b", 0.6),
        ("qwen3_1p7b_base_fr_dev100.json", "qwen3_1p7b", 1.7),
    ],
    "zh": [
        ("qwen3_0p6b_base_zh_dev100.json", "qwen3_0p6b", 0.6),
        ("qwen3_1p7b_base_zh_dev100.json", "qwen3_1p7b", 1.7),
    ],
}


QWEN_FINETUNES = {
    "fr": [
        ("qwen3_0p6b_lora_fr_dev100.json", "qwen3_0p6b", "+ LoRA (dev100)"),
        ("qwen3_0p6b_full_fr_dev100.json", "qwen3_0p6b", "+ full FT (dev100)"),
        ("qwen3_1p7b_full_fr_dev100.json", "qwen3_1p7b", "+ full FT (dev100)"),
    ],
    "zh": [
        ("qwen3_0p6b_full_zh_dev100.json", "qwen3_0p6b", "+ full FT (dev100)"),
        ("qwen3_1p7b_full_zh_dev100.json", "qwen3_1p7b", "+ full FT (dev100)"),
    ],
}


def load_qwen_baselines(language: str) -> list[dict]:
    rows = []
    for filename, tag_root, params_b in QWEN_BASELINES[language]:
        path = os.path.join(QWEN_PREDS_DIR, filename)
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "tag_root": tag_root,
            "name": DISPLAY[tag_root],
            "params_b": params_b,
            "wer": r.get("wer"),
            "cer": r.get("cer"),
        })
    return rows


# Map fine-tune-tag → (parent baseline tag_root in our data, label for the FT).
FT_PARENT_FR = {
    "lora_tiny":      ("tiny",   "+ LoRA (zh recipe)"),
    "full_tiny":      ("tiny",   "+ full FT"),
    "lora_small":     ("small",  "+ LoRA (zh recipe)"),
    "lora_small_v2":  ("small",  "+ LoRA (fr recipe)"),
    "full_small":     ("small",  "+ full FT"),
    "scratch_small":  ("small",  "from scratch"),
    "lora_medium":    ("medium", "+ LoRA"),
    "lora_turbo":     ("turbo",  "+ LoRA"),
}
FT_PARENT_ZH = {
    "lora_tiny_zh": ("tiny",   "+ LoRA (zh recipe)"),
    "full_tiny_zh": ("tiny",   "+ full FT"),
    "lora_small":  ("small",  "+ LoRA (zh recipe)"),
    "full_small":  ("small",  "+ full FT"),
    "lora_medium": ("medium", "+ LoRA (zh recipe)"),
}


def load_finetunes(metrics_path: str, parent_map: dict, metric_key: str) -> list[dict]:
    with open(metrics_path) as f:
        d = json.load(f)
    out = []
    for r in d["rows"]:
        tag = r["tag"]
        if tag in parent_map:
            parent_tag, ft_label = parent_map[tag]
            out.append({
                "tag": tag,
                "ft_label": ft_label,
                "parent_tag": parent_tag,
                "wer": r.get("wer"),
                "cer": r.get("cer"),
            })
    return out


def load_qwen_finetunes(language: str) -> list[dict]:
    out = []
    metric_key = "wer" if language == "fr" else "cer"
    for filename, parent_tag, ft_label in QWEN_FINETUNES[language]:
        path = os.path.join(QWEN_PREDS_DIR, filename)
        with open(path) as f:
            r = json.load(f)
        out.append({
            "tag": filename.replace(".json", ""),
            "ft_label": ft_label,
            "parent_tag": parent_tag,
            "wer": r.get("wer"),
            "cer": r.get("cer"),
        })
    return out


def finetune_style(ft: dict) -> tuple[str, str]:
    label = ft.get("ft_label", "").lower()
    tag = ft.get("tag", "").lower()
    if "lora" in label or tag.startswith("lora_"):
        return LORA_COLOR, "#9a5412"
    return FULL_FT_COLOR, "#4a1022"


# ---------- Drawing ----------


def make_figure(rows, finetunes, metric_key, metric_label, title, out_path,
                language):
    rows = [r for r in rows if r["params_b"] is not None and r[metric_key] is not None]
    if language == "fr":
        rows = [r for r in rows if r["tag_root"] != "canary_qwen"]
        finetunes = [ft for ft in finetunes if ft["tag"] != "scratch_small"]
    rows.sort(key=lambda r: r["params_b"])
    parents_by_tag = {r["tag_root"]: r for r in rows}

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(16, 8.8))

    xs = np.array([r["params_b"] for r in rows])
    ys = np.array([r[metric_key] * 100 for r in rows])
    ft_vals = [ft.get(metric_key) * 100 for ft in finetunes if ft.get(metric_key) is not None]
    all_y = np.array(list(ys) + ft_vals)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (Billions)", fontweight="bold", fontsize=18)
    ax.set_ylabel(f"{metric_label} (%)", fontweight="bold", fontsize=18)
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold")
    ax.grid(True, which="both", axis="y", linestyle="-", linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=15)

    ax.set_xlim(0.03, 30)
    ymax = Y_LIMITS.get(language, all_y.max() + 4.0)
    ax.set_ylim(0, ymax)

    # 1. Baseline dots / external references.
    for r, x, y in zip(rows, xs, ys):
        color = IN_SCOPE_COLOR if r["tag_root"] in FT_SCOPE_TAGS else OUT_SCOPE_COLOR
        ax.scatter([x], [y], s=220, c=color, edgecolors="#444",
                   linewidths=0.9, zorder=3)

    # 2. Fine-tune arrows + endpoint markers. Stagger x-jitter per FT so
    # multiple fine-tunes of the same baseline don't stack on top of each other.
    seen_per_parent: dict = {}
    JITTER_FACTORS = [1.04, 1.10, 1.18, 0.92, 0.86]
    for ft in finetunes:
        parent = parents_by_tag.get(ft["parent_tag"])
        if parent is None:
            continue
        x0 = parent["params_b"]
        y0 = parent[metric_key] * 100
        v = ft.get(metric_key)
        if v is None:
            continue
        y1 = v * 100
        idx = seen_per_parent.get(ft["parent_tag"], 0)
        seen_per_parent[ft["parent_tag"]] = idx + 1
        x1 = x0 * JITTER_FACTORS[idx % len(JITTER_FACTORS)]
        marker_color, edge_color = finetune_style(ft)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=ARROW_COLOR,
                lw=2.2,
                shrinkA=5, shrinkB=5,
                connectionstyle="arc3,rad=0.10",
            ),
            zorder=2,
        )
        if y1 <= ymax:
            ax.scatter([x1], [y1], s=120, marker="s", c=marker_color,
                       edgecolors=edge_color, linewidths=1.2, zorder=4)

    # 3. Baseline labels.
    OFF = OFFSETS_FR if language == "fr" else OFFSETS_ZH
    for r, x, y in zip(rows, xs, ys):
        dx, dy, ha = OFF.get(r["tag_root"], (14, 0, "left"))
        ax.annotate(
            r["name"], xy=(x, y),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=13, color="#202020", ha=ha, fontweight="medium",
            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.8,
                            shrinkA=4, shrinkB=2),
        )

    # 4. Legend (point classes only).
    scope_h = plt.Line2D([0], [0], marker="o", linestyle="None",
                         markerfacecolor=IN_SCOPE_COLOR, markeredgecolor="#444",
                         markersize=11, label="Baseline in fine-tune scope")
    other_h = plt.Line2D([0], [0], marker="o", linestyle="None",
                         markerfacecolor=OUT_SCOPE_COLOR, markeredgecolor="#444",
                         markersize=11, label="Baseline not fine-tuned here")
    lora_h = plt.Line2D([0], [0], marker="s", linestyle="None",
                        markerfacecolor=LORA_COLOR, markeredgecolor="#9a5412",
                        markersize=10, label="LoRA")
    full_h = plt.Line2D([0], [0], marker="s", linestyle="None",
                        markerfacecolor=FULL_FT_COLOR, markeredgecolor="#4a1022",
                        markersize=10, label="Full fine-tune")
    ax.legend(handles=[scope_h, other_h, lora_h, full_h], loc="upper right",
              fontsize=13, frameon=True, framealpha=0.96)

    fig.text(0.99, 0.97,
             "speech2text/asr_bench + README rows + Qwen dev100 fine-tunes",
             ha="right", va="top", fontsize=11, color="#888", style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    print("saved", out_path, "(+ .pdf)")


# ---------- Main ----------


def main():
    fr_rows = collect_baselines("fr")
    zh_rows = collect_baselines("zh")
    fr_rows.extend(load_extra_readme_baselines("fr"))
    fr_rows.extend(load_qwen_baselines("fr"))
    zh_rows.extend(load_qwen_baselines("zh"))
    fr_ft = load_finetunes(FR_METRICS, FT_PARENT_FR, "wer")
    zh_ft = load_finetunes(ZH_METRICS, FT_PARENT_ZH, "cer")
    fr_ft.extend(load_qwen_finetunes("fr"))
    zh_ft.extend(load_qwen_finetunes("zh"))

    print("--- fr-FR baselines ---")
    for r in sorted(fr_rows, key=lambda r: r["params_b"] or 0):
        wer = f"{r['wer']*100:.2f}%" if r["wer"] is not None else "—"
        print(f"  {r['tag_root']:<22} {r['params_b']:>6.2f} B  WER={wer}")
    print("--- fr-FR fine-tunes (Δ vs parent baseline) ---")
    for ft in fr_ft:
        parent = next(r for r in fr_rows if r["tag_root"] == ft["parent_tag"])
        delta = (ft["wer"] - parent["wer"]) * 100
        print(f"  {ft['tag']:<18} parent={ft['parent_tag']:<10} "
              f"WER={ft['wer']*100:>6.2f}%  Δ={delta:+.2f}pt")

    print("--- zh-CN baselines ---")
    for r in sorted(zh_rows, key=lambda r: r["params_b"] or 0):
        cer = f"{r['cer']*100:.2f}%" if r["cer"] is not None else "—"
        print(f"  {r['tag_root']:<22} {r['params_b']:>6.2f} B  CER={cer}")
    print("--- zh-CN fine-tunes (Δ vs parent baseline) ---")
    for ft in zh_ft:
        parent = next(r for r in zh_rows if r["tag_root"] == ft["parent_tag"])
        delta = (ft["cer"] - parent["cer"]) * 100
        print(f"  {ft['tag']:<18} parent={ft['parent_tag']:<10} "
              f"CER={ft['cer']*100:>6.2f}%  Δ={delta:+.2f}pt")

    if fr_rows:
        make_figure(
            fr_rows, fr_ft, "wer", "Word Error Rate",
            "WER vs Model Size — FLEURS fr-FR",
            os.path.join(FIG_DIR, "wer_vs_size_fr.png"),
            language="fr",
        )
    if zh_rows:
        make_figure(
            zh_rows, zh_ft, "cer", "Character Error Rate",
            "CER vs Model Size — Common Voice 21 zh-CN",
            os.path.join(FIG_DIR, "wer_vs_size_zh.png"),
            language="zh",
        )


if __name__ == "__main__":
    main()
