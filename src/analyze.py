"""Build the results table + bucket worst errors by linguistic category for French.

Reads:  outputs/preds/<tag>.json files from src/eval.py
Writes: outputs/metrics.json (table), outputs/error_analysis.json (worst errors)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import List

from jiwer import cer, wer


# ---- French-specific error categorization helpers ----

# Common homophones in French (same pronunciation, different spelling/grammar).
# We list a small representative set; comprehensive coverage isn't the goal.
HOMOPHONES = {
    ("a", "à"), ("à", "a"),
    ("ou", "où"), ("où", "ou"),
    ("son", "sont"), ("sont", "son"),
    ("ces", "ses"), ("ses", "ces"), ("ces", "c'est"), ("c'est", "ces"),
    ("est", "et"), ("et", "est"),
    ("ce", "se"), ("se", "ce"),
    ("la", "là"), ("là", "la"),
    ("on", "ont"), ("ont", "on"),
    ("mes", "mais"), ("mais", "mes"),
    ("près", "prêt"), ("prêt", "près"),
    ("peu", "peux"), ("peux", "peu"), ("peu", "peut"), ("peut", "peu"),
    ("vert", "verre"), ("verre", "vert"),
    ("foie", "foi"), ("foi", "foie"),
}

LATIN_RE = re.compile(r"[A-Za-z]+")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _diff(ref: str, hyp: str):
    import difflib

    sm = difflib.SequenceMatcher(a=ref.split(), b=hyp.split())
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            ops.append((tag, " ".join(ref.split()[i1:i2]), " ".join(hyp.split()[j1:j2])))
    return ops


def categorize(ref: str, hyp: str) -> List[str]:
    cats = []
    if not ref.strip():
        return ["empty_ref"]
    if not hyp.strip():
        return ["empty_hyp"]

    if len(hyp) > 1.5 * len(ref):
        cats.append("hallucination_long")
    if len(hyp) < 0.5 * len(ref):
        cats.append("truncation")

    edits = _diff(ref, hyp)
    for op, r, h in edits:
        if op == "replace":
            r_strip = _strip_accents(r)
            h_strip = _strip_accents(h)
            if r_strip == h_strip and r != h:
                cats.append("accent_only")
            elif (r, h) in HOMOPHONES or (h, r) in HOMOPHONES:
                cats.append("homophone")
            elif _strip_accents(r).rstrip("s") == _strip_accents(h).rstrip("s"):
                # plural / number agreement (ses → ce, livres → livre, etc.)
                cats.append("agreement_or_plural")
            elif _strip_accents(r).rstrip("e") == _strip_accents(h).rstrip("e"):
                # masculine vs feminine
                cats.append("agreement_or_plural")
            else:
                cats.append("word_substitution")
        if op == "insert":
            cats.append("insertion")
        if op == "delete":
            cats.append("deletion")

    return cats or ["other"]


def per_example_wer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return wer([ref], [hyp])


def per_example_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return cer([ref], [hyp])


def _infer_language(d: dict, fname: str) -> str:
    """Determine pred language from explicit field, else from filename suffix.

    Old preds (pre-tiny work) don't have a `language` field. Conventions:
    - tag ending with `_zh` → zh
    - file living under archive_zh/preds → zh
    - else → fr
    """
    if d.get("language") in ("fr", "zh"):
        return d["language"]
    tag = fname[:-5]  # strip .json
    if tag.endswith("_zh") or "/archive_zh/" in fname:
        return "zh"
    return "fr"


def build_table(preds_dir: str, language_filter: str | None = None,
                extra_dirs: list[str] | None = None) -> List[dict]:
    rows = []
    seen_tags = set()
    paths = []
    for d in [preds_dir] + (extra_dirs or []):
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".json"):
                paths.append(os.path.join(d, fname))
    for path in paths:
        fname = os.path.basename(path)
        with open(path) as f:
            d = json.load(f)
        lang = _infer_language(d, path)
        if language_filter is not None and lang != language_filter:
            continue
        tag = fname[:-5]
        if tag in seen_tags:
            continue  # earlier preds_dir wins (outputs/preds before archive_zh)
        seen_tags.add(tag)
        rows.append(
            {
                "tag": tag,
                "model_id": d.get("model_id"),
                "adapter": d.get("adapter"),
                "split": d.get("split"),
                "language": lang,
                "n": d.get("n"),
                "wer": None if d.get("wer") is None else round(d.get("wer"), 4),
                "cer": None if d.get("cer") is None else round(d.get("cer"), 4),
                "wall_seconds": round(d.get("wall_seconds", 0), 1),
            }
        )
    return rows


def deltas(rows: List[dict], primary: str = "wer") -> List[dict]:
    """Add abs/rel deltas on `primary` vs the matching baseline.

    Tag matching: a baseline is identified by its 'baseline_<size>' tag. Other
    rows match by *containing* the size as a token, e.g. 'lora_small_v2' or
    'full_small' both match 'small'. We prefer the longest matching size so
    that 'lora_medium' doesn't accidentally hit 'small' if both are present.

    `primary` is "wer" for fr, "cer" for zh — it's the column on which we
    compute the delta.
    """
    abs_key = f"{primary}_abs_delta"
    rel_key = f"{primary}_rel_delta_pct"
    baselines = {}
    for r in rows:
        if r["tag"].startswith("baseline_"):
            size = r["tag"][len("baseline_"):]
            baselines[size] = r[primary]
    out = []
    # Longest size first so e.g. 'tiny_zh' beats 'tiny' on `lora_tiny_zh`.
    sized = sorted(baselines.keys(), key=len, reverse=True)
    for r in rows:
        r2 = dict(r)
        if r["tag"].startswith("baseline_"):
            out.append(r2)
            continue
        match = None
        for size in sized:
            # Match if tag ends with _<size> or contains <size> as a token.
            if r["tag"].endswith("_" + size) or ("_" + size + "_") in ("_" + r["tag"] + "_"):
                match = size
                break
        if match is not None:
            b = baselines[match]
            v = r[primary]
            if v is not None and b is not None:
                r2[abs_key] = round(v - b, 4)
                r2[rel_key] = round(100 * (v - b) / b, 2) if b else None
        out.append(r2)
    return out


def bucket_errors(preds_dir: str, focus_tag: str, top_k: int = 100) -> dict:
    path = os.path.join(preds_dir, f"{focus_tag}.json")
    with open(path) as f:
        d = json.load(f)
    items = []
    for ex in d["predictions"]:
        w = per_example_wer(ex["ref_norm"], ex["hyp_norm"])
        c = per_example_cer(ex["ref_norm"], ex["hyp_norm"])
        cats = categorize(ex["ref_norm"], ex["hyp_norm"])
        items.append({"wer": w, "cer": c, "cats": cats, **ex})
    items.sort(key=lambda x: x["wer"], reverse=True)
    worst = items[:top_k]

    cat_counter: Counter = Counter()
    examples_by_cat = defaultdict(list)
    for it in worst:
        for c in it["cats"]:
            cat_counter[c] += 1
            if len(examples_by_cat[c]) < 5:
                examples_by_cat[c].append(
                    {"ref": it["ref_norm"], "hyp": it["hyp_norm"], "wer": round(it["wer"], 3)}
                )
    return {
        "focus": focus_tag,
        "n_worst": len(worst),
        "cat_counts": dict(cat_counter),
        "examples_by_cat": dict(examples_by_cat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="/data/speech2text/outputs/preds")
    ap.add_argument("--out-dir", default="/data/speech2text/outputs")
    ap.add_argument("--language", choices=["fr", "zh"], default="fr",
                    help="Filter preds by language and pick the primary metric")
    ap.add_argument("--out-name", default=None,
                    help="metrics filename (default: metrics_<lang>.json)")
    ap.add_argument("--focus-tag", default=None,
                    help="Pred tag to bucket errors on. Defaults to whichever has lowest primary metric.")
    args = ap.parse_args()

    primary = "wer" if args.language == "fr" else "cer"
    extra_dirs = ["/data/speech2text/archive_zh/preds"] if args.language == "zh" else None
    rows = build_table(args.preds_dir, language_filter=args.language, extra_dirs=extra_dirs)
    rows = deltas(rows, primary=primary)

    if not args.focus_tag:
        candidates = [r for r in rows if r[primary] is not None and r[primary] < 0.99]
        if candidates:
            args.focus_tag = min(candidates, key=lambda r: r[primary])["tag"]
        elif rows:
            args.focus_tag = min(rows, key=lambda r: r[primary] or 1.0)["tag"]

    metrics = {"language": args.language, "rows": rows}
    out_name = args.out_name or f"metrics_{args.language}.json"
    with open(os.path.join(args.out_dir, out_name), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"wrote {os.path.join(args.out_dir, out_name)}")
    for r in rows:
        wer = "-" if r.get("wer") is None else f"{r['wer']:.4f}"
        cer = "-" if r.get("cer") is None else f"{r['cer']:.4f}"
        ad = r.get(f"{primary}_abs_delta", "-")
        rd = r.get(f"{primary}_rel_delta_pct", "-")
        print(f"  {r['tag']:<28} WER={wer} CER={cer} abs_delta={ad} rel%={rd}")

    if args.focus_tag and args.language == "fr":
        ea = bucket_errors(args.preds_dir, args.focus_tag, top_k=100)
        ea_name = f"error_analysis_{args.language}.json"
        with open(os.path.join(args.out_dir, ea_name), "w") as f:
            json.dump(ea, f, ensure_ascii=False, indent=2)
        print(f"wrote {os.path.join(args.out_dir, ea_name)} (focus={args.focus_tag})")
        print("category counts:", ea["cat_counts"])


if __name__ == "__main__":
    main()
