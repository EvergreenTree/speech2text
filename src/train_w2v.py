"""Train a CTC head + adapter on wav2vec2 / XLS-R for French ASR on FLEURS fr_fr.

This is the *alternate-paradigm* comparison: instead of Whisper's seq2seq + huge
pretrained vocabulary, we use a self-supervised acoustic encoder + a CTC head
defined on a French character vocabulary built from the training transcripts.
Same compute budget, very different inductive biases.

We start from a French-fine-tuned XLS-R (`bhuang/asr-wav2vec2-french`) so the
encoder already knows about French phonotactics — the goal is to test whether
this paradigm can match / beat Whisper-LoRA at the same compute.

Single GPU (L4 23 GB), bf16, ~5-10 min wall.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


SR = 16_000


def build_vocab(texts: List[str]) -> dict:
    """Char vocab: lowercase French letters, accents, apostrophe, space, [PAD], [UNK]."""
    chars = set()
    for t in texts:
        chars.update(t)
    # We always include the standard French letters + accented + apostrophe + space
    # so the vocab is stable across splits.
    forced = set("abcdefghijklmnopqrstuvwxyzéèêëàâäîïôöùûüçñœæ' ")
    chars = sorted(chars | forced)
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = vocab.pop(" ")  # Wav2Vec2 convention: word-separator is "|"
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


@dataclass
class W2VCollator:
    processor: Wav2Vec2Processor
    pad_token_id: int = -100

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), self.pad_token_id
        )
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-dev", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--freeze-feature-encoder", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    raw = load_dataset("google/fleurs", "fr_fr", trust_remote_code=True)
    raw = raw.cast_column("audio", Audio(sampling_rate=SR))
    if "validation" in raw and "dev" not in raw:
        raw["dev"] = raw["validation"]

    if args.n_train is not None:
        raw["train"] = raw["train"].shuffle(seed=args.seed).select(range(min(args.n_train, len(raw["train"]))))
    if args.n_dev is not None:
        raw["dev"] = raw["dev"].shuffle(seed=args.seed).select(range(min(args.n_dev, len(raw["dev"]))))

    vocab = build_vocab(raw["train"]["transcription"])
    vocab_path = os.path.join(args.out, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SR, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.out)

    def encode(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        text = batch["transcription"]
        text = text.replace(" ", "|")
        with processor.as_target_processor():
            batch["labels"] = processor(text).input_ids
        return batch

    proc = DatasetDict()
    for split in ["train", "dev"]:
        proc[split] = raw[split].map(encode, remove_columns=raw[split].column_names, num_proc=4)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=25,
        save_steps=2000,
        save_total_limit=2,
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=proc["train"],
        data_collator=W2VCollator(processor=processor),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
