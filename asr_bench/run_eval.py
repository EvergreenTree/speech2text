"""Unified ASR inference + WER/CER for FLEURS-fr and CV21-zh test slices.

Supports four model families:
  - whisper      : openai/whisper-* and any Whisper-arch fine-tune (Belle, distil-fr).
  - voxtral      : mistralai/Voxtral-* via VoxtralForConditionalGeneration.
  - sensevoice   : FunAudioLLM/SenseVoiceSmall via AutoModel(trust_remote_code).
  - wav2vec2-ctc : encoder-only + CTC head (e.g. bofenghuang/asr-wav2vec2-ctc-french).

Reads test_{lang}.pkl produced by prep_data.py. Writes preds and metrics into
preds/<tag>.json. Idempotent: skips if the prediction file already exists.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import time
import unicodedata

# IMPORTANT: keep all caches off /data which is full.
os.environ.setdefault("HF_HUB_CACHE", "/data/speech2text/asr_bench/cache")
os.environ.setdefault("HF_HOME", "/data/speech2text/asr_bench/cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/speech2text/outputs/cache")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/speech2text/asr_bench/cache")

import torch
from jiwer import cer, wer

OUT_DIR = "/data/speech2text/asr_bench"
PREDS_DIR = os.path.join(OUT_DIR, "preds")
os.makedirs(PREDS_DIR, exist_ok=True)


# -------- Normalization (matches /data/speech2text/src/eval.py) ----------

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


def normalize_text(text: str, language: str) -> str:
    return normalize_zh(text) if language == "zh" else normalize_fr(text)


# -------- Audio loading ----------


def load_test(lang: str) -> list:
    path = os.path.join(OUT_DIR, f"test_{lang}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# -------- Model adapters ----------


def whisper_inference(model_id: str, items: list, language: str, dtype, batch_size: int):
    from transformers import (
        AutoProcessor,
        WhisperForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model.to("cuda").eval()

    n_params = sum(p.numel() for p in model.parameters())
    sr = items[0]["sr"]

    forced_ids = processor.tokenizer.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.max_length = None

    preds = []
    t0 = time.time()
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        feats = processor.feature_extractor(
            [c["audio"] for c in chunk],
            sampling_rate=sr,
            return_tensors="pt",
        ).input_features
        feats = feats.to("cuda", dtype=dtype)
        with torch.inference_mode():
            ids = model.generate(
                feats,
                forced_decoder_ids=forced_ids,
                num_beams=1,
                max_new_tokens=225,
                do_sample=False,
            )
        preds.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        print(f"  whisper {start + len(chunk)}/{len(items)}", flush=True)
    elapsed = time.time() - t0

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return preds, n_params, elapsed


def voxtral_inference(model_id: str, items: list, language: str, dtype, batch_size: int,
                      use_4bit: bool = False):
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "cuda"
    else:
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = "cuda"
    model = VoxtralForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    sr = items[0]["sr"]

    preds = []
    t0 = time.time()
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        # Voxtral exposes a transcription helper (apply_transcription_request).
        audio_list = [c["audio"] for c in chunk]
        inputs = processor.apply_transcription_request(
            language=language,
            audio=audio_list,
            sampling_rate=sr,
            format=["WAV"] * len(audio_list),
            model_id=model_id,
            return_tensors="pt",
        )
        # Move to GPU; cast only the audio mel tensor to the model dtype.
        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if v.dtype.is_floating_point:
                    moved[k] = v.to("cuda", dtype=dtype)
                else:
                    moved[k] = v.to("cuda")
            else:
                moved[k] = v
        with torch.inference_mode():
            # 256 tokens easily covers any FLEURS-fr (avg ~24 words / ~80 BPE tokens)
            # or CV21-zh utterance and stops early on EOS.
            ids = model.generate(**moved, max_new_tokens=256, do_sample=False)
        # Strip the prompt tokens; processor decodes only the new ones.
        prompt_lens = moved["input_ids"].shape[1]
        new_ids = ids[:, prompt_lens:]
        decoded = processor.batch_decode(new_ids, skip_special_tokens=True)
        preds.extend(decoded)
        print(f"  voxtral {start + len(chunk)}/{len(items)}", flush=True)
    elapsed = time.time() - t0

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return preds, n_params, elapsed


def sensevoice_inference(model_id: str, items: list, language: str, dtype, batch_size: int):
    """SenseVoice-Small via FunASR's AutoModel.

    The HuggingFace `transformers` AutoModel doesn't expose a clean ASR head, so
    we use the official funasr inference loop.
    """
    from funasr import AutoModel as FunASRModel

    # SenseVoice's funasr loader downloads to its own cache; redirect that too.
    os.environ["MODELSCOPE_CACHE"] = "/data/speech2text/asr_bench/cache/modelscope"
    # FunASR resolves model IDs against ModelScope; map the HF mirror to the
    # ModelScope canonical id.
    ms_id = model_id
    if model_id.startswith("FunAudioLLM/SenseVoiceSmall"):
        ms_id = "iic/SenseVoiceSmall"
    model = FunASRModel(model=ms_id, device="cuda", trust_remote_code=True,
                        disable_update=True)
    sr = items[0]["sr"]

    # Estimate params from the underlying torch module if exposed.
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        n_params = -1

    lang_code = "zh" if language == "zh" else "auto"
    preds = []
    t0 = time.time()
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        out = model.generate(
            input=[c["audio"] for c in chunk],
            cache={},
            language=lang_code,
            use_itn=True,
            batch_size_s=60,
            sampling_rate=sr,
        )
        # FunASR returns a list with {"key", "text"}. The text contains
        # SenseVoice tags like <|en|><|HAPPY|><|Speech|>; strip them.
        for r in out:
            t = r["text"]
            t = re.sub(r"<\|[^|]+\|>", "", t)
            preds.append(t.strip())
        print(f"  sensevoice {start + len(chunk)}/{len(items)}", flush=True)
    elapsed = time.time() - t0

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds, n_params, elapsed


def wav2vec2_ctc_inference(model_id: str, items: list, language: str, dtype, batch_size: int):
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id, torch_dtype=dtype).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    sr = items[0]["sr"]

    preds = []
    t0 = time.time()
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        inputs = processor(
            [c["audio"] for c in chunk],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        feats = inputs.input_values.to("cuda", dtype=dtype)
        with torch.inference_mode():
            logits = model(feats).logits
        ids = logits.argmax(dim=-1)
        decoded = processor.batch_decode(ids)
        preds.extend(decoded)
        print(f"  w2v2 {start + len(chunk)}/{len(items)}", flush=True)
    elapsed = time.time() - t0

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return preds, n_params, elapsed


def voxtral_4bit_inference(model_id, items, language, dtype, batch_size):
    return voxtral_inference(model_id, items, language, dtype, batch_size, use_4bit=True)


FAMILIES = {
    "whisper": whisper_inference,
    "voxtral": voxtral_inference,
    "voxtral_4bit": voxtral_4bit_inference,
    "sensevoice": sensevoice_inference,
    "wav2vec2-ctc": wav2vec2_ctc_inference,
}


# -------- Driver ----------


def evaluate(tag: str, model_id: str, family: str, language: str, lang_profile: str,
             batch_size: int, dtype_name: str, force: bool = False):
    out_path = os.path.join(PREDS_DIR, f"{tag}.json")
    if os.path.exists(out_path) and not force:
        print(f"[{tag}] already exists, skipping. (use --force to redo)")
        return

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    items = load_test(lang_profile)
    fn = FAMILIES[family]
    print(f"[{tag}] family={family} model={model_id} n={len(items)} dtype={dtype_name}")
    preds, n_params, elapsed = fn(model_id, items, language, dtype, batch_size)

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
        "tag": tag,
        "model_id": model_id,
        "family": family,
        "language": language,
        "lang_profile": lang_profile,
        "n": len(items),
        "n_params": n_params,
        "wer": wer_score,
        "cer": cer_score,
        "wall_seconds": elapsed,
        "predictions": [
            {"ref": it["ref"], "hyp": p, "ref_norm": rn, "hyp_norm": pn}
            for it, p, rn, pn in zip(items, preds, refs_norm, preds_norm)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    if wer_score is not None:
        print(
            f"[{tag}] WER={wer_score:.4f} CER={cer_score:.4f} "
            f"params={n_params / 1e6:.0f}M sec={elapsed:.1f}"
        )
    else:
        print(
            f"[{tag}] CER={cer_score:.4f} "
            f"params={n_params / 1e6:.0f}M sec={elapsed:.1f}"
        )
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=list(FAMILIES.keys()))
    ap.add_argument("--language", required=True, choices=["fr", "zh"])
    ap.add_argument("--lang-profile", required=True, choices=["fleurs_fr", "cv21_zh"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    evaluate(args.tag, args.model, args.family, args.language, args.lang_profile,
             args.batch_size, args.dtype, args.force)


if __name__ == "__main__":
    main()
