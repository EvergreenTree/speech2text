import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
    "to_q",
    "to_kv",
    "to_out",
    "linear",
]


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prompt(language: str) -> str:
    return f"<|audio|>transcribe the audio in {language} to text."


def make_preprocess_fn():
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "audio": ex["audio"],
            "target": ex["reference"],
            "language": ex["language"],
            "prompt_text": build_prompt(ex["language"]),
        }

    return _preprocess


@dataclass
class DataCollatorForGraniteSpeechFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]
        targets = [f["target"] for f in features]
        audios = [load_audio(path, sr=self.sampling_rate) for path in audio_paths]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [f"{prompt}{target}{eos}" for prompt, target in zip(prompt_texts, targets)]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prompt_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, prefix_len in enumerate(prefix_lens):
            labels[i, :prefix_len] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(dtype=model_dtype)
        return inputs


def build_lora_model(model, rank: int, alpha: int, dropout: float):
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def count_trainable_parameters(model) -> Dict[str, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"trainable": trainable, "total": total}


def enable_gradient_checkpointing_fallback(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return
        except ValueError:
            pass

    inner_lm = getattr(model, "language_model", None)
    if inner_lm is not None and hasattr(inner_lm, "gradient_checkpointing_enable"):
        inner_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(inner_lm, "enable_input_require_grads"):
            inner_lm.enable_input_require_grads()


def parse_args():
    p = argparse.ArgumentParser("Granite Speech Finetuning")
    p.add_argument("--model_path", type=str, default="ibm-granite/granite-speech-4.1-2b")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--mode", type=str, default="full", choices=["full", "lora"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", type=str, default="")
    return p.parse_args()


def resolve_resume_checkpoint(output_dir: str, explicit: str) -> str | None:
    if explicit:
        return explicit

    root = Path(output_dir)
    if not root.exists():
        return None

    checkpoints = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, child))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def main():
    args = parse_args()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=None,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.config.use_cache = False

    if args.gradient_checkpointing == 1:
        enable_gradient_checkpointing_fallback(model)

    if args.mode == "lora":
        model = build_lora_model(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    counts = count_trainable_parameters(model)
    print(
        f"[params] mode={args.mode} trainable={counts['trainable']:,} "
        f"total={counts['total']:,} pct={counts['trainable'] / counts['total'] * 100:.2f}%"
    )

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            **({"validation": args.eval_file} if args.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn(), num_proc=1)
    keep = {"audio", "target", "language", "prompt_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForGraniteSpeechFinetuning(processor=processor, sampling_rate=args.sr)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=(args.pin_memory == 1),
        dataloader_persistent_workers=(args.persistent_workers == 1 and args.num_workers > 0),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=bool(args.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        processing_class=processor,
    )
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_checkpoint:
        print(f"[resume] checkpoint={resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "lora":
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
