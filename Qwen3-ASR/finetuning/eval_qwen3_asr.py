import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import librosa
import torch
from jiwer import cer, mer, wer
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
from qwen_asr.inference.utils import parse_asr_output

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

PUNCT_RE = re.compile(r"[\.\!\?,\:\;\"«»\(\)\[\]\{\}—…·–‐\-]+")
APOSTROPHE_BOUNDARY_RE = re.compile(r"(^|\s)['’]+|['’]+(\s|$)")
SMART_QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"'}
WS_RE = re.compile(r"\s+")
CJK_PUNCT_RE = re.compile(r"[，。！？、；：「」『』（）《》〈〉—…·　—,\.\!\?,\:\;\"'\(\)\[\]\{\}\-]+")


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
    if language == "Chinese":
        return normalize_zh(text)
    return normalize_fr(text)


def build_messages(audio_payload):
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
    ]


def build_prompt(processor, forced_language: str | None):
    prompt = processor.apply_chat_template(build_messages(""), add_generation_prompt=True, tokenize=False)
    if forced_language:
        prompt = prompt + f"language {forced_language}<asr_text>"
    return prompt


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model(model_path: str, adapter_path: str | None, dtype):
    model = AutoModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=None,
        trust_remote_code=False,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True, trust_remote_code=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, processor


def transcribe_batch(model, processor, rows, batch_size: int, forced_language: str | None, dtype):
    device = next(model.parameters()).device
    prompt = build_prompt(processor, forced_language)
    preds = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        audios = [librosa.load(item["audio"], sr=16000, mono=True)[0] for item in batch]
        prompts = [prompt] * len(batch)
        inputs = processor(text=prompts, audio=audios, return_tensors="pt", padding=True)
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                if value.is_floating_point():
                    inputs[key] = value.to(device=device, dtype=dtype)
                else:
                    inputs[key] = value.to(device=device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        continuation = sequences[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for text in decoded:
            _, parsed = parse_asr_output(text, user_language=forced_language)
            preds.append(parsed)
    return preds


def evaluate(rows, preds, language: str):
    refs = [row["reference"] for row in rows]
    refs_norm = [normalize_text(ref, language) for ref in refs]
    preds_norm = [normalize_text(pred, language) for pred in preds]
    safe_refs = [x if x else "<empty>" for x in refs_norm]
    safe_preds = [x if x else "<empty>" for x in preds_norm]

    if language == "Chinese":
        return {
            "wer": None,
            "cer": cer(safe_refs, safe_preds),
            "mer": mer([" ".join(list(x)) for x in safe_refs], [" ".join(list(x)) for x in safe_preds]),
            "predictions": [
                {"ref": ref, "hyp": hyp, "ref_norm": rn, "hyp_norm": pn}
                for ref, hyp, rn, pn in zip(refs, preds, refs_norm, preds_norm)
            ],
        }

    return {
        "wer": wer(safe_refs, safe_preds),
        "cer": cer(safe_refs, safe_preds),
        "mer": None,
        "predictions": [
            {"ref": ref, "hyp": hyp, "ref_norm": rn, "hyp_norm": pn}
            for ref, hyp, rn, pn in zip(refs, preds, refs_norm, preds_norm)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/data/speech2text/Qwen3-ASR/finetuning/outputs/preds")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--force-language", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    rows = load_jsonl(args.jsonl)
    language = rows[0]["language"]
    model, processor = load_model(args.model_path, args.adapter_path, dtype)

    t0 = time.time()
    preds = transcribe_batch(
        model=model,
        processor=processor,
        rows=rows,
        batch_size=args.batch_size,
        forced_language=args.force_language,
        dtype=dtype,
    )
    metrics = evaluate(rows, preds, language)
    elapsed = time.time() - t0

    result = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "jsonl": args.jsonl,
        "tag": args.tag,
        "language": language,
        "n": len(rows),
        "wall_seconds": elapsed,
        **metrics,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    if result["wer"] is None:
        print(f"[{args.tag}] CER={result['cer']:.4f} MER={result['mer']:.4f} n={result['n']} sec={result['wall_seconds']:.1f}")
    else:
        print(f"[{args.tag}] WER={result['wer']:.4f} CER={result['cer']:.4f} n={result['n']} sec={result['wall_seconds']:.1f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
