"""Inference + WER/CER for Whisper baseline / fine-tuned / from-scratch checkpoints
on FLEURS fr_fr.

Usage:
    python -m src.eval --model openai/whisper-small --tag baseline_small \
        --raw-dir /data/speech2text/outputs/cache/processed/openai__whisper-small/raw \
        --split test --batch-size 16

Optionally pass --adapter <path> to load a PEFT/LoRA adapter on top of `--model`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from typing import List

import torch
from datasets import load_from_disk
from jiwer import cer, mer, wer
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)


# Punctuation we drop before scoring. We keep apostrophes inside words because
# French elision (l'enfant, qu'il, c'est) is part of the orthographic word.
PUNCT_RE = re.compile(r"[\.\!\?,\:\;\"«»\(\)\[\]\{\}—…·–‐\-]+")
# Trailing/leading apostrophes (used as quotes) are noise; we leave intra-word
# apostrophes alone.
APOSTROPHE_BOUNDARY_RE = re.compile(r"(^|\s)['’]+|['’]+(\s|$)")
SMART_QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"'}
WS_RE = re.compile(r"\s+")

# Chinese: CJK punct + Latin punct + whitespace all drop. No tokenization
# (Mandarin orthographic words are not whitespace-delimited).
CJK_PUNCT_RE = re.compile(r"[，。！？、；：「」『』（）《》〈〉—…·　—,\.\!\?,\:\;\"'\(\)\[\]\{\}\-]+")


def normalize_fr(text: str) -> str:
    """Light French normalization for fair scoring.

    - NFC-normalize Unicode (combine accents into a single codepoint).
    - Map smart quotes to ASCII so liaison apostrophes are consistent.
    - Lowercase.
    - Drop sentence punctuation but keep intra-word apostrophes/hyphens that
      are part of the lexical form (mot à mot → mot à mot, but j'ai → j'ai).
    - Collapse whitespace.

    We deliberately do NOT strip accents: in French, é/è/à/ô are lexical, not
    decorative — confusing 'a' with 'à' (preposition vs verb) is a real error.
    """
    text = unicodedata.normalize("NFC", text)
    for k, v in SMART_QUOTES.items():
        text = text.replace(k, v)
    text = text.lower()
    text = APOSTROPHE_BOUNDARY_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


def normalize_zh(text: str) -> str:
    """Drop punctuation/whitespace before computing CER on Chinese.

    Whisper output sometimes adds punctuation that is not in the CV ground
    truth, or misses punctuation that is. We strip both sides for a fair
    character-level comparison. Fullwidth + halfwidth punctuation is removed.
    """
    text = CJK_PUNCT_RE.sub("", text)
    text = WS_RE.sub("", text)
    return text.strip()


def normalize_text(text: str, language: str) -> str:
    if language == "zh":
        return normalize_zh(text)
    return normalize_fr(text)


def load_model(model_id: str, adapter: str | None, device: str, dtype):
    base = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel

        base = PeftModel.from_pretrained(base, adapter)
        base = base.merge_and_unload()  # fold LoRA into base for fast inference
    base.to(device).eval()
    return base


def transcribe(
    model,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    audio_arrays: List,
    sampling_rate: int,
    device,
    dtype,
    batch_size: int = 8,
    num_beams: int = 1,
    language: str = "fr",
) -> List[str]:
    forced_ids = tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.forced_decoder_ids = None
    preds: List[str] = []
    n = len(audio_arrays)
    for start in range(0, n, batch_size):
        chunk = audio_arrays[start : start + batch_size]
        inputs = feature_extractor(
            chunk, sampling_rate=sampling_rate, return_tensors="pt"
        )
        feats = inputs.input_features.to(device=device, dtype=dtype)
        with torch.inference_mode():
            ids = model.generate(
                feats,
                forced_decoder_ids=forced_ids,
                num_beams=num_beams,
                max_new_tokens=225,
                do_sample=False,
            )
        preds.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return preds


def evaluate_split(
    model_id: str,
    raw_dir: str,
    split: str,
    adapter: str | None,
    device: str,
    dtype,
    batch_size: int,
    num_beams: int,
    language: str,
    text_field: str,
) -> dict:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_id, language=language, task="transcribe"
    )
    model = load_model(model_id, adapter, device, dtype)

    ds = load_from_disk(raw_dir)[split]
    audio = [ex["audio"]["array"] for ex in ds]
    # Pick first available text field — different mirrors use different names.
    candidates = [text_field, "transcription", "sentence", "raw_transcription"]
    chosen = next((c for c in candidates if c in ds.column_names), None)
    if chosen is None:
        raise KeyError(f"None of {candidates} present in raw dataset columns: {ds.column_names}")
    refs = [ex[chosen] for ex in ds]
    sr = ds[0]["audio"]["sampling_rate"]

    t0 = time.time()
    preds = transcribe(
        model,
        feature_extractor,
        tokenizer,
        audio,
        sr,
        device,
        dtype,
        batch_size=batch_size,
        num_beams=num_beams,
        language=language,
    )
    elapsed = time.time() - t0

    refs_norm = [normalize_text(r, language) for r in refs]
    preds_norm = [normalize_text(p, language) for p in preds]

    # Guard: jiwer's WER chokes on empty refs; FLEURS shouldn't have any but
    # the from-scratch model produces empty strings often.
    safe_refs = [r if r else "<empty>" for r in refs_norm]
    safe_preds = [p if p else "<empty>" for p in preds_norm]

    if language == "zh":
        # Chinese: no whitespace word boundaries. WER is degenerate on raw
        # joined text, so we space-join characters to make WER comparable to
        # CER but still account for ins/del separately. CER stays primary.
        wer_score = None
        cer_score = cer(safe_refs, safe_preds)
        ref_chars = [" ".join(list(r)) for r in safe_refs]
        pred_chars = [" ".join(list(p)) for p in safe_preds]
        mer_score = mer(ref_chars, pred_chars)
    else:
        wer_score = wer(safe_refs, safe_preds)
        cer_score = cer(safe_refs, safe_preds)
        mer_score = None

    return {
        "model_id": model_id,
        "adapter": adapter,
        "split": split,
        "language": language,
        "n": len(refs),
        "wer": wer_score,
        "cer": cer_score,
        "mer": mer_score,
        "wall_seconds": elapsed,
        "predictions": [
            {"ref": r, "hyp": p, "ref_norm": rn, "hyp_norm": pn}
            for r, p, rn, pn in zip(refs, preds, refs_norm, preds_norm)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True, help="filename tag for outputs/preds/<tag>.json")
    ap.add_argument("--out-dir", default="/data/speech2text/outputs/preds")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--language", default="fr")
    ap.add_argument("--text-field", default="transcription",
                    help="raw column to use as ground truth (FLEURS exposes 'transcription' and 'raw_transcription').")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = evaluate_split(
        model_id=args.model,
        raw_dir=args.raw_dir,
        split=args.split,
        adapter=args.adapter,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        language=args.language,
        text_field=args.text_field,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    if res['wer'] is not None:
        print(f"[{args.tag}] WER={res['wer']:.4f} CER={res['cer']:.4f} n={res['n']} sec={res['wall_seconds']:.1f}")
    else:
        print(f"[{args.tag}] CER={res['cer']:.4f} MER={res['mer']:.4f} n={res['n']} sec={res['wall_seconds']:.1f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
