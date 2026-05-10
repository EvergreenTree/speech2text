"""Run NVIDIA NeMo ASR models (Parakeet-TDT, Canary-Qwen) and write a
prediction JSON in the same shape as run_eval.py.

Lives in /data/nemo_venv (separate venv — NeMo pins transformers 4.57 which
collides with the main env's transformers 5.6).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import pickle
import time
import unicodedata

# Cache on /data
os.environ.setdefault("HF_HUB_CACHE", "/data/speech2text/asr_bench/cache")
os.environ.setdefault("HF_HOME", "/data/speech2text/asr_bench/cache")

import numpy as np
import torch
import soundfile as sf
import tempfile
from jiwer import cer, wer

OUT_DIR = "/data/speech2text/asr_bench"
PREDS_DIR = os.path.join(OUT_DIR, "preds")
os.makedirs(PREDS_DIR, exist_ok=True)


# --- Same normalization as run_eval.py ---

PUNCT_RE = re.compile(r"[\.\!\?,\:\;\"«»\(\)\[\]\{\}—…·–‐\-]+")
APOSTROPHE_BOUNDARY_RE = re.compile(r"(^|\s)['’]+|['’]+(\s|$)")
SMART_QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"'}
WS_RE = re.compile(r"\s+")
CJK_PUNCT_RE = re.compile(
    r"[，。！？、；：「」『』（）《》〈〉—…·　—,\.\!\?,\:\;\"'\(\)\[\]\{\}\-]+"
)


def normalize_fr(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for k, v in SMART_QUOTES.items():
        text = text.replace(k, v)
    text = text.lower()
    text = APOSTROPHE_BOUNDARY_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


def normalize_zh(text: str) -> str:
    text = CJK_PUNCT_RE.sub("", text)
    text = WS_RE.sub("", text)
    return text.strip()


def normalize_text(t: str, language: str) -> str:
    return normalize_zh(t) if language == "zh" else normalize_fr(t)


def write_temp_wav(audio: np.ndarray, sr: int, scratch_dir: str, idx: int) -> str:
    path = os.path.join(scratch_dir, f"{idx:04d}.wav")
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def run_parakeet(model_id: str, items: list, language: str, batch_size: int):
    """Multilingual Parakeet-TDT (e.g. nvidia/parakeet-tdt-0.6b-v3).

    Loaded via nemo_asr.ASRModel.from_pretrained → .transcribe([wav_paths]).
    """
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cuda")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    sr = items[0]["sr"]
    preds = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        wav_paths = [write_temp_wav(it["audio"], sr, scratch, i) for i, it in enumerate(items)]
        # Multilingual variants accept source_lang / language hint via .transcribe.
        try:
            out = model.transcribe(
                audio=wav_paths,
                batch_size=batch_size,
                source_lang=language,
                target_lang=language,
                pnc="no",
                timestamps=False,
                verbose=False,
            )
        except TypeError:
            # Older signature without language flags.
            out = model.transcribe(audio=wav_paths, batch_size=batch_size, verbose=False)
    elapsed = time.time() - t0

    # NeMo returns a list of Hypothesis objects with .text attr (or strings).
    preds = []
    for h in out:
        if hasattr(h, "text"):
            preds.append(h.text)
        elif isinstance(h, (list, tuple)) and h:
            preds.append(getattr(h[0], "text", str(h[0])))
        else:
            preds.append(str(h))
    return preds, n_params, elapsed


def run_canary(model_id: str, items: list, language: str, batch_size: int):
    """Canary-Qwen-2.5B (SALM) — SALM.generate with chat-style prompts.

    The SALM model exposes ``generate(prompts, audios, audio_lens, ...)``,
    not ``.transcribe``. We feed pre-loaded float32 audios (B, T) zero-padded
    along the time axis with explicit per-sample lengths.
    """
    from nemo.collections.speechlm2.models import SALM
    import numpy as np
    import torch as _torch

    model = SALM.from_pretrained(model_id).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    audio_locator_tag = model.audio_locator_tag
    sr = items[0]["sr"]
    assert sr == model.sampling_rate, f"sr mismatch {sr} vs {model.sampling_rate}"

    # Canary-Qwen auto-detects the audio language; do NOT pass an "in <lang>"
    # hint — that triggers translation rather than transcription.
    instr = f"Transcribe the following: {audio_locator_tag}"

    preds = []
    t0 = time.time()
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        max_t = max(len(c["audio"]) for c in chunk)
        audios = _torch.zeros((len(chunk), max_t), dtype=_torch.float32)
        for i, c in enumerate(chunk):
            arr = c["audio"]
            audios[i, : len(arr)] = _torch.from_numpy(arr)
        audio_lens = _torch.tensor([len(c["audio"]) for c in chunk],
                                   dtype=_torch.int64)
        prompts = [[{"role": "user", "content": instr}] for _ in chunk]
        with _torch.inference_mode():
            ids = model.generate(
                prompts=prompts,
                audios=audios.to(model.device),
                audio_lens=audio_lens.to(model.device),
                max_new_tokens=256,
                do_sample=False,
            )
        # ids returned by SALM.generate are the *answer* (not the prompt) tokens.
        # NeMo's AutoTokenizer uses ids_to_text(list[int]) per row.
        ids_cpu = ids.detach().cpu().tolist()
        for row in ids_cpu:
            # Strip pad/eos.
            eos_id = model.tokenizer.eos_id
            pad_id = model.tokenizer.pad_id
            cleaned = [t for t in row if t not in (eos_id, pad_id)]
            txt = model.tokenizer.ids_to_text(cleaned)
            txt = txt.replace("<|im_end|>", "").replace("<|im_start|>assistant", "")
            txt = txt.lstrip("assistant ").lstrip("\n").strip()
            preds.append(txt)
        print(f"  canary {start + len(chunk)}/{len(items)}", flush=True)
    elapsed = time.time() - t0
    return preds, n_params, elapsed


FAMILIES = {
    "parakeet": run_parakeet,
    "canary":   run_canary,
}


def evaluate(tag: str, model_id: str, family: str, language: str,
             lang_profile: str, batch_size: int, force: bool = False):
    out_path = os.path.join(PREDS_DIR, f"{tag}.json")
    if os.path.exists(out_path) and not force:
        print(f"[{tag}] already exists, skipping.")
        return

    test_path = os.path.join(OUT_DIR, f"test_{lang_profile}.pkl")
    with open(test_path, "rb") as f:
        items = pickle.load(f)

    fn = FAMILIES[family]
    print(f"[{tag}] family={family} model={model_id} n={len(items)}")
    preds, n_params, elapsed = fn(model_id, items, language, batch_size)

    refs_norm = [normalize_text(it["ref"], language) for it in items]
    preds_norm = [normalize_text(p, language) for p in preds]
    safe_refs = [r if r else "<empty>" for r in refs_norm]
    safe_preds = [p if p else "<empty>" for p in preds_norm]

    if language == "zh":
        cer_score = cer(safe_refs, safe_preds)
        wer_score = None
    else:
        wer_score = wer(safe_refs, safe_preds)
        cer_score = cer(safe_refs, safe_preds)

    res = {
        "tag": tag, "model_id": model_id, "family": family,
        "language": language, "lang_profile": lang_profile,
        "n": len(items), "n_params": n_params,
        "wer": wer_score, "cer": cer_score,
        "wall_seconds": elapsed,
        "predictions": [
            {"ref": it["ref"], "hyp": p, "ref_norm": rn, "hyp_norm": pn}
            for it, p, rn, pn in zip(items, preds, refs_norm, preds_norm)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    if wer_score is not None:
        print(f"[{tag}] WER={wer_score:.4f} CER={cer_score:.4f} params={n_params/1e6:.0f}M sec={elapsed:.1f}")
    else:
        print(f"[{tag}] CER={cer_score:.4f} params={n_params/1e6:.0f}M sec={elapsed:.1f}")
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=list(FAMILIES.keys()))
    ap.add_argument("--language", required=True, choices=["fr", "zh"])
    ap.add_argument("--lang-profile", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    evaluate(args.tag, args.model, args.family, args.language,
             args.lang_profile, args.batch_size, args.force)


if __name__ == "__main__":
    main()
