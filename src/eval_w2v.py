"""Inference + WER/CER for a wav2vec2 CTC model on FLEURS fr_fr."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.eval import normalize_fr


SR = 16_000


def transcribe_w2v(model, processor, audio_arrays: List, device, dtype, batch_size: int = 8) -> List[str]:
    preds = []
    n = len(audio_arrays)
    for start in range(0, n, batch_size):
        chunk = audio_arrays[start : start + batch_size]
        inp = processor(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
        feats = inp.input_values.to(device=device, dtype=dtype)
        attn = inp.attention_mask.to(device=device) if "attention_mask" in inp else None
        with torch.inference_mode():
            logits = model(feats, attention_mask=attn).logits
        pred_ids = torch.argmax(logits, dim=-1)
        preds.extend(processor.batch_decode(pred_ids))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to trained model dir or HF id")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/data/speech2text/outputs/preds")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500, help="cap on test examples for parity with Whisper eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    raw = load_dataset("google/fleurs", "fr_fr", split=args.split, trust_remote_code=True)
    raw = raw.cast_column("audio", Audio(sampling_rate=SR))
    if args.n is not None and len(raw) > args.n:
        raw = raw.shuffle(seed=args.seed).select(range(args.n))

    audio = [ex["audio"]["array"] for ex in raw]
    refs = [ex["transcription"] for ex in raw]

    t0 = time.time()
    preds = transcribe_w2v(model, processor, audio, device, dtype, batch_size=args.batch_size)
    elapsed = time.time() - t0

    refs_norm = [normalize_fr(r) for r in refs]
    preds_norm = [normalize_fr(p) for p in preds]
    safe_refs = [r if r else "<empty>" for r in refs_norm]
    safe_preds = [p if p else "<empty>" for p in preds_norm]
    res = {
        "model_id": args.model,
        "adapter": None,
        "split": args.split,
        "n": len(refs),
        "wer": wer(safe_refs, safe_preds),
        "cer": cer(safe_refs, safe_preds),
        "wall_seconds": elapsed,
        "predictions": [
            {"ref": r, "hyp": p, "ref_norm": rn, "hyp_norm": pn}
            for r, p, rn, pn in zip(refs, preds, refs_norm, preds_norm)
        ],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[{args.tag}] WER={res['wer']:.4f} CER={res['cer']:.4f} n={res['n']} sec={elapsed:.1f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
