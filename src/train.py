"""Training script: LoRA, full FT, or random-init from-scratch — for Whisper on
FLEURS fr_fr.

Modes:
- lora    : freeze base, train PEFT LoRA adapters on q/k/v/out_proj.
- full    : train all weights of the pretrained base.
- scratch : random-init the same architecture (via from_config), then train all weights.
            Tokenizer and feature extractor still come from the pretrained id, since the
            vocabulary / mel-spectrogram conventions are part of the data pipeline.

Examples:
    python -m src.train --model openai/whisper-small --mode lora \
        --processed-dir /data/speech2text/outputs/cache/processed/openai__whisper-small/processed \
        --out outputs/adapters/lora_small \
        --batch 16 --grad-accum 1 --epochs 1 --lr 1e-4

    python -m src.train --model openai/whisper-small --mode scratch \
        --processed-dir ... --out outputs/adapters/scratch_small \
        --batch 16 --grad-accum 1 --epochs 10 --lr 5e-4 --warmup-ratio 0.05
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from src.data import WhisperDataCollator


# Standard Whisper LoRA recipe — all attention projections, encoder + decoder.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]


def build_lora(model, rank: int, alpha: int, dropout: float):
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type=None,  # no preset; Whisper isn't in PEFT's task enum
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["lora", "full", "scratch"], required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="fr")

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Full FT only: freeze encoder, train decoder only.")
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(args.model, language=args.language, task="transcribe")

    if args.mode == "scratch":
        cfg = WhisperConfig.from_pretrained(args.model)
        model = WhisperForConditionalGeneration(cfg)
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_pretrained(args.model)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model)

    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    if args.mode == "lora":
        model = build_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    elif args.mode == "full" and args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    processed = load_from_disk(args.processed_dir)
    collator = WhisperDataCollator(feature_extractor=feature_extractor, tokenizer=tokenizer)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=False,  # we toggle on the model directly
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=processed["train"],
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()

    if args.mode == "lora":
        model.save_pretrained(args.out)
    else:
        trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    feature_extractor.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
