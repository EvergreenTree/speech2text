"""Gradio server: record/upload audio, transcribe with baseline + LoRA-FT side by side.

Run:
    python -m src.server \
        --base openai/whisper-small \
        --lora /data/speech2text/outputs/adapters/lora_small \
        --port 7860 \
        --host 0.0.0.0

Open http://<server-ip>:7860 in your browser.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)


SR = 16_000


class WhisperEngine:
    def __init__(self, model_id: str, adapter: Optional[str], device: str, dtype):
        self.model_id = model_id
        self.adapter = adapter
        self.device = device
        self.dtype = dtype
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_id, language="fr", task="transcribe"
        )
        model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        self.model = model.to(device).eval()
        self.forced_ids = self.tokenizer.get_decoder_prompt_ids(
            language="fr", task="transcribe"
        )

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int, num_beams: int = 1) -> str:
        if sr != SR:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
            sr = SR
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        feats = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        x = feats.input_features.to(device=self.device, dtype=self.dtype)
        ids = self.model.generate(
            x,
            forced_decoder_ids=self.forced_ids,
            num_beams=num_beams,
            max_new_tokens=225,
            do_sample=False,
        )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def build_app(baseline: WhisperEngine, finetuned: Optional[WhisperEngine], ft_label: str):
    desc_lines = [
        f"**Baseline:** `{baseline.model_id}`",
    ]
    if finetuned:
        adapter_str = finetuned.adapter or "(full FT)"
        desc_lines.append(f"**Fine-tuned ({ft_label}):** `{finetuned.model_id}` + `{adapter_str}`")
    else:
        desc_lines.append("_(no fine-tuned model loaded — only baseline)_")
    desc = "\n\n".join(desc_lines)

    def transcribe(audio, num_beams):
        if audio is None:
            return "", "", ""
        sr, arr = audio
        # Mic returns int16, file uploads can be int16/int32/float.
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
            sr = SR

        t0 = time.time()
        b = baseline.transcribe(arr, sr, num_beams=int(num_beams))
        t1 = time.time()
        if finetuned:
            f = finetuned.transcribe(arr, sr, num_beams=int(num_beams))
            t2 = time.time()
            timing = f"baseline {t1 - t0:.2f}s | fine-tuned {t2 - t1:.2f}s | audio {len(arr)/sr:.2f}s"
        else:
            f = ""
            timing = f"baseline {t1 - t0:.2f}s | audio {len(arr)/sr:.2f}s"
        return b, f, timing

    with gr.Blocks(title="Whisper fr — baseline vs fine-tuné") as app:
        gr.Markdown("# Whisper fr : baseline vs fine-tuné")
        gr.Markdown(desc)
        with gr.Row():
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Parlez en français ou uploadez un fichier (wav/mp3/flac/ogg)",
                streaming=False,
                interactive=True,
            )
        with gr.Row():
            num_beams = gr.Slider(1, 5, value=1, step=1, label="num_beams (1 = greedy)")
            btn = gr.Button("Transcrire", variant="primary")
        with gr.Row():
            base_out = gr.Textbox(label=f"Baseline ({baseline.model_id.split('/')[-1]})", lines=3)
            ft_out = gr.Textbox(label=f"Fine-tuné ({ft_label})", lines=3)
        timing_out = gr.Markdown()
        btn.click(transcribe, inputs=[audio, num_beams], outputs=[base_out, ft_out, timing_out])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--lora", default=None, help="Path to LoRA adapter dir")
    ap.add_argument("--full", default=None, help="Path to full-FT model dir")
    ap.add_argument("--ft-label", default="LoRA")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading baseline {args.base} ...")
    baseline = WhisperEngine(args.base, adapter=None, device=device, dtype=dtype)

    finetuned = None
    label = args.ft_label
    if args.full:
        print(f"loading full-FT model {args.full} ...")
        finetuned = WhisperEngine(args.full, adapter=None, device=device, dtype=dtype)
        label = label or "Full FT"
    elif args.lora:
        print(f"loading baseline + LoRA {args.lora} ...")
        finetuned = WhisperEngine(args.base, adapter=args.lora, device=device, dtype=dtype)

    app = build_app(baseline, finetuned, ft_label=label)
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
