# coding=utf-8
"""
MWER (Minimum Word Error Rate) fine-tuning for Qwen3-ASR.

Algorithm (Prabhavalkar et al. 2018):
  1. Generate N-best hypotheses via beam search on the current model (no grad).
  2. Re-score the N-best with teacher-forcing forward passes (grad flows here).
     The raw generated text (including lang tag) is scored, not a re-tokenised
     clean copy — this ensures P̂(y_i|x) is for the sequence that was actually
     produced.
  3. Renormalize over the N-best support  ->  P_hat(y_i | x).
  4. MWER loss = sum_i P_hat(y_i|x) * (W_i - W_bar)
  5. CE on ground truth (weight lambda_ce) for stability.

Key correctness choices:
  - padding_side="right" for scoring inputs so prefix_lens are not displaced
    by left-pad tokens (flipped back to "left" for generation).
  - Loss computed on raw hypothesis sequences, WER on normalised parsed text.
"""
import argparse
import contextlib
import gc
import json
import math
import os
import random
import re
import sys
import time
import types
import unicodedata
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
from jiwer import cer, wer
from transformers import (AutoConfig, AutoModel, AutoProcessor,
                          GenerationConfig, get_linear_schedule_with_warmup)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr.core.transformers_backend import (Qwen3ASRConfig,
                                                Qwen3ASRForConditionalGeneration,
                                                Qwen3ASRProcessor)
from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
    _get_feat_extract_output_lengths,
)
from qwen_asr.inference.utils import parse_asr_output

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

# ── Text normalisation ────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[\.\!\?,\:\;\"«»\(\)\[\]\{\}—…·–‐\-]+")
_APOSTROPHE_RE = re.compile(r"(^|\s)['']+|['']+(\s|$)")
_SMART = {"'": "'", "'": "'", "\u201c": '"', "\u201d": '"'}
_WS_RE = re.compile(r"\s+")
_CJK_PUNCT_RE = re.compile(
    r"[，。！？、；：「」『』（）《》〈〉—…·　—,\.\!\?,\:\;\"'\(\)\[\]\{\}\-]+"
)


def _norm_fr(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    for k, v in _SMART.items():
        t = t.replace(k, v)
    t = t.lower()
    t = _APOSTROPHE_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _norm_zh(t: str) -> str:
    return _WS_RE.sub("", _CJK_PUNCT_RE.sub("", t)).strip()


def normalise(text: str, language: str) -> str:
    return _norm_zh(text) if language == "Chinese" else _norm_fr(text)


def compute_error_rate(hyp: str, ref: str, language: str) -> float:
    hyp_n, ref_n = normalise(hyp, language), normalise(ref, language)
    if not ref_n:
        return 0.0
    sr, sh = ref_n or "<empty>", hyp_n or "<empty>"
    return cer(sr, sh) if language == "Chinese" else wer(sr, sh)


# ── Model helpers ─────────────────────────────────────────────────────────────

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    def forward(self, input_ids=None, attention_mask=None,
                 input_features=None, feature_attention_mask=None,
                 labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def patch_embedding_access(model):
    if hasattr(model, "thinker") and hasattr(model.thinker, "model"):
        model.get_input_embeddings = types.MethodType(
            lambda self: self.thinker.model.get_input_embeddings(), model)
        model.set_input_embeddings = types.MethodType(
            lambda self, v: self.thinker.model.set_input_embeddings(v), model)


def enable_input_require_grads(model):
    try:
        model.enable_input_require_grads()
        return
    except NotImplementedError:
        pass
    for attr in ("model", "thinker"):
        inner = getattr(model, attr, None)
        if inner and hasattr(inner, "get_input_embeddings"):
            embed = inner.get_input_embeddings()
            if embed is not None:
                embed.register_forward_hook(
                    lambda m, i, o: o.requires_grad_(True))
                return


def cast_inputs(inputs, dtype, device):
    return {
        k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) and v.is_floating_point()
            else v.to(device=device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


@contextlib.contextmanager
def generation_cache_enabled(*models):
    saved = []
    for model in models:
        entries = {"cache": [], "gradient_checkpointing": []}
        objs = [model, getattr(model, "thinker", None)]
        thinker = getattr(model, "thinker", None)
        if thinker is not None:
            objs.append(getattr(thinker, "model", None))
        seen = set()
        for obj in objs:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            cfg = getattr(obj, "config", None)
            if cfg is not None and hasattr(cfg, "use_cache"):
                entries["cache"].append((cfg, cfg.use_cache))
                cfg.use_cache = True
            if (hasattr(obj, "gradient_checkpointing_disable")
                    and getattr(obj, "is_gradient_checkpointing", False)):
                obj.gradient_checkpointing_disable()
                entries["gradient_checkpointing"].append(obj)
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                entries["gradient_checkpointing"].append(
                    (module, module.gradient_checkpointing))
                module.gradient_checkpointing = False
        saved.append(entries)
    try:
        yield
    finally:
        for entries in saved:
            for cfg, value in entries["cache"]:
                cfg.use_cache = value
            for obj in entries["gradient_checkpointing"]:
                if isinstance(obj, tuple):
                    module, value = obj
                    module.gradient_checkpointing = value
                else:
                    obj.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})


def build_audio_cache(processor, audios: list) -> dict:
    cache = processor.feature_extractor(
        audios,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )
    return {
        "input_features": cache["input_features"],
        "feature_attention_mask": cache["attention_mask"],
    }


def processor_with_cached_audio(processor, texts: list, audio_cache: dict,
                                audio_indices: list, return_tensors="pt",
                                padding=True, truncation=False) -> dict:
    feature_mask = audio_cache["feature_attention_mask"]
    audio_lengths = _get_feat_extract_output_lengths(feature_mask.sum(-1))
    expanded_lengths = [int(audio_lengths[i].item()) for i in audio_indices]
    processed_texts = processor.replace_multimodal_special_tokens(
        list(texts), iter(expanded_lengths))
    text_inputs = processor.tokenizer(
        processed_texts,
        return_tensors=return_tensors,
        padding=padding,
        truncation=truncation,
    )
    idx = torch.tensor(audio_indices, dtype=torch.long)
    return {
        **text_inputs,
        "input_features": audio_cache["input_features"].index_select(0, idx),
        "feature_attention_mask": feature_mask.index_select(0, idx),
    }


# ── Vectorised sequence scoring ───────────────────────────────────────────────

_SCORE_KEYS = ("input_ids", "attention_mask", "input_features", "feature_attention_mask")
_SCORE_ROW_CHUNK = int(os.environ.get("QWEN_ASR_SCORE_ROW_CHUNK", "2"))


def compute_sequence_logp(model, inputs: dict, prefix_lens: list,
                           pad_id: int, no_grad: bool = False) -> torch.Tensor:
    """
    Return the summed per-token log-prob of hypothesis tokens for each sequence.

    Assumes RIGHT-padded inputs so that token[n, plen:] is the hypothesis for
    sequence n (call processor with padding_side='right' before this function).

    Args:
        model       : patched Qwen3ASRForConditionalGeneration
        inputs      : right-padded processor output, on correct device/dtype
        prefix_lens : [N] int, the length of the prompt portion per sequence
        pad_id      : tokenizer pad id
        no_grad     : run under torch.no_grad()
    Returns:
        logp_sums : [N] float tensor (grad-capable when no_grad=False)
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        fwd = {k: inputs[k] for k in _SCORE_KEYS if k in inputs}
        out = model(**fwd)
        logits = out.logits          # [N, T, V]  (model dtype)
        N, T, V = logits.shape
        input_ids = inputs["input_ids"]   # [N, T]

        logp_parts = []
        positions = torch.arange(T - 1, device=input_ids.device).unsqueeze(0)
        for lo in range(0, N, _SCORE_ROW_CHUNK):
            hi = min(lo + _SCORE_ROW_CHUNK, N)
            shifted_ids = input_ids[lo:hi, 1:].clamp(min=0)
            shifted_logp = F.log_softmax(logits[lo:hi, :-1], dim=-1)
            gathered = shifted_logp.gather(
                2, shifted_ids.unsqueeze(-1)).squeeze(-1).float()

            plen_t = torch.tensor(prefix_lens[lo:hi], device=input_ids.device).unsqueeze(1)
            hyp_mask = ((positions >= (plen_t - 1))
                        & (input_ids[lo:hi, 1:] != pad_id))
            logp_parts.append((gathered * hyp_mask.float()).sum(dim=1))
        logp_sums = torch.cat(logp_parts, dim=0)
    return logp_sums


# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prefix_text(processor) -> str:
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    return processor.apply_chat_template([msgs], add_generation_prompt=True, tokenize=False)[0]


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, processor, rows, language: str, batch_size: int,
             dtype, device, forced_language: str, max_rows: int = 100) -> dict:
    rows = rows[:max_rows]
    prompt_base = build_prefix_text(processor)
    prompt_with_lang = (prompt_base + f"language {forced_language}<asr_text>"
                        if forced_language else prompt_base)

    # Generation needs left-padding
    orig_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    preds, refs = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        audios  = [librosa.load(r["audio"], sr=16000, mono=True)[0] for r in batch]
        prompts = [prompt_with_lang] * len(batch)
        inputs  = processor(text=prompts, audio=audios, return_tensors="pt", padding=True)
        inputs  = cast_inputs(inputs, dtype, device)
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        seqs = gen.sequences if hasattr(gen, "sequences") else gen
        cont = seqs[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(cont, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        for text, row in zip(decoded, batch):
            _, parsed = parse_asr_output(text, user_language=forced_language)
            preds.append(parsed)
            refs.append(row["reference"])

    processor.tokenizer.padding_side = orig_side

    refs_n  = [normalise(r, language) for r in refs]
    preds_n = [normalise(p, language) for p in preds]
    sr = [x or "<empty>" for x in refs_n]
    sp = [x or "<empty>" for x in preds_n]

    result: dict = {"n": len(rows), "language": language}
    if language == "Chinese":
        result["wer"] = None
        result["cer"] = cer(sr, sp)
    else:
        result["wer"] = wer(sr, sp)
        result["cer"] = cer(sr, sp)
    return result


def save_eval(metrics, args, out_path):
    metrics["tag"]        = args.tag
    metrics["model_path"] = args.model_path
    metrics["jsonl"]      = args.eval_file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if metrics["wer"] is None:
        print(f"  CER={metrics['cer']:.4f}  n={metrics['n']}")
    else:
        print(f"  WER={metrics['wer']:.4f}  CER={metrics['cer']:.4f}  n={metrics['n']}")
    print(f"  saved {out_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_mwer(args):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype  = torch.bfloat16 if use_bf16 else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(
        args.model_path, dtype=dtype, device_map=None, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(
        args.model_path, fix_mistral_regex=True, trust_remote_code=False)

    patch_outer_forward(model)
    patch_embedding_access(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        enable_input_require_grads(model)

    model.to(device).train()

    pad_id  = processor.tokenizer.pad_token_id or 0
    eos_str = processor.tokenizer.eos_token or ""

    train_rows = load_jsonl(args.train_file)
    eval_rows  = load_jsonl(args.eval_file) if args.eval_file else []
    prefix_text = build_prefix_text(processor)
    lang_tag    = f"language {args.language}<asr_text>"

    # Exact step limit for fractional epochs
    max_samples = int(len(train_rows) * args.epochs)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    accum_target = max(args.grad_acc, args.mwer_batch_size)
    total_steps  = math.ceil(max_samples / accum_target)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    samples_seen = 0
    accum_samples = 0
    optimizer.zero_grad()
    t0 = time.time()

    while samples_seen < max_samples:
        random.shuffle(train_rows)
        for start in range(0, len(train_rows), args.mwer_batch_size):
            if samples_seen >= max_samples:
                break
            batch = train_rows[start:start + args.mwer_batch_size]
            remaining = max_samples - samples_seen
            batch = batch[:remaining]
            if not batch:
                break
            batch_size = len(batch)
            samples_seen += batch_size
            accum_samples += batch_size

            audios = [librosa.load(r["audio"], sr=16000, mono=True)[0] for r in batch]
            audio_cache = build_audio_cache(processor, audios)

            # ── 1. Generate N-best (left-pad for generation) ───────────────────
            processor.tokenizer.padding_side = "left"
            gen_inputs = processor_with_cached_audio(
                processor,
                texts=[prefix_text] * batch_size,
                audio_cache=audio_cache,
                audio_indices=list(range(batch_size)),
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            gen_inputs = cast_inputs(gen_inputs, dtype, device)
            prompt_width = gen_inputs["input_ids"].shape[1]

            gen_kwargs = {
                "max_new_tokens": 256,
                "num_return_sequences": args.n_best,
            }
            if args.generation_strategy == "sample":
                gen_kwargs.update(
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=1,
                )
            else:
                gen_kwargs.update(
                    do_sample=False,
                    num_beams=args.n_best,
                    early_stopping=True,
                )

            with generation_cache_enabled(model):
                with torch.no_grad():
                    gen_out = model.generate(
                        **gen_inputs,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        **gen_kwargs)
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
            # [B*N, T_full], grouped by source sample.

            # Decode raw continuations (keep lang tag intact for scoring)
            raw_hyps = []     # [B][N], raw decoded text used for scoring
            clean_hyps = []   # [B][N], parsed transcript used for WER
            for b in range(batch_size):
                raw_row, clean_row = [], []
                for n in range(args.n_best):
                    seq_i = b * args.n_best + n
                    cont_ids = sequences[seq_i, prompt_width:]
                    raw = processor.decode(
                        cont_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                    _, parsed = parse_asr_output(raw, user_language=None)
                    raw_row.append(raw)
                    clean_row.append(parsed)
                raw_hyps.append(raw_row)
                clean_hyps.append(clean_row)

            # ── 2. WER per hypothesis (parsed text vs reference) ───────────────
            W = torch.tensor(
                [[compute_error_rate(h, row["reference"], args.language)
                  for h in hyps]
                 for row, hyps in zip(batch, clean_hyps)],
                device=device, dtype=torch.float32)  # [B,N]

            # ── 3. Score raw hypotheses with grad (right-pad for correct offset) ──
            # Score: prefix_text + raw_hyp + eos  (raw already contains lang tag)
            processor.tokenizer.padding_side = "right"
            flat_audio_indices = [b for b in range(batch_size) for _ in range(args.n_best)]
            flat_raw_hyps = [h for hyps in raw_hyps for h in hyps]
            full_texts = [prefix_text + h + eos_str for h in flat_raw_hyps]
            prefix_texts = [prefix_text] * len(full_texts)

            full_inputs = processor_with_cached_audio(
                processor, full_texts, audio_cache, flat_audio_indices,
                return_tensors="pt", padding=True, truncation=False)
            # Compute prefix length once per sample (all N hyps share the same prefix)
            prefix_inputs_1 = processor_with_cached_audio(
                processor, [prefix_text] * batch_size, audio_cache, list(range(batch_size)),
                return_tensors="pt", padding=True, truncation=False)
            full_inputs     = cast_inputs(full_inputs,     dtype, device)
            prefix_inputs_1 = cast_inputs(prefix_inputs_1, dtype, device)
            plen_per_sample = prefix_inputs_1["attention_mask"].sum(dim=1).tolist()  # [B]
            prefix_lens     = [plen_per_sample[b]
                                for b in range(batch_size)
                                for _ in range(args.n_best)]                          # [B*N]

            logp = compute_sequence_logp(model, full_inputs, prefix_lens, pad_id,
                                          no_grad=False).view(batch_size, args.n_best)

            # ── 4. MWER loss ───────────────────────────────────────────────────
            p_hat  = F.softmax(logp, dim=1)    # [B,N]
            W_bar  = W.mean(dim=1, keepdim=True)
            L_mwer_per_sample = (p_hat * (W - W_bar)).sum(dim=1)
            L_mwer = L_mwer_per_sample.mean()
            mwer_value = float(L_mwer.detach().item())

            mwer_loss = L_mwer * (batch_size / accum_target)
            mwer_loss.backward()
            del (full_inputs, prefix_inputs_1, logp, p_hat, W_bar,
                 L_mwer_per_sample, L_mwer, mwer_loss)
            cleanup_cuda()

            # ── 5. CE on ground truth (stability) ─────────────────────────────
            # Always: prefix_text + "language X<asr_text>" + reference + eos
            ce_targets = [lang_tag + row["reference"] + eos_str for row in batch]
            ce_full = processor_with_cached_audio(
                processor,
                texts=[prefix_text + t for t in ce_targets],
                audio_cache=audio_cache,
                audio_indices=list(range(batch_size)),
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            ce_full  = cast_inputs(ce_full, dtype, device)
            # Reuse plen_per_sample (same prefix, same audio) — no extra processor call
            plen_gt  = plen_per_sample
            n_gt     = (ce_full["attention_mask"].sum(dim=1) -
                        torch.tensor(plen_gt, device=device)).clamp(min=1).float()
            logp_gt = compute_sequence_logp(model, ce_full, plen_gt, pad_id,
                                            no_grad=False)  # [B]
            L_ce = (-logp_gt / n_gt).mean()
            ce_value = float(L_ce.detach().item())

            ce_loss = (args.lambda_ce * L_ce) * (batch_size / accum_target)
            ce_loss.backward()
            batch_loss_value = mwer_value + args.lambda_ce * ce_value
            del (ce_full, logp_gt, n_gt, L_ce, ce_loss, W, gen_inputs,
                 gen_out, sequences, audio_cache)
            cleanup_cuda()

            # ── 6. Optimiser step ──────────────────────────────────────────────
            if accum_samples >= accum_target or samples_seen == max_samples:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_samples = 0

                if global_step % args.log_steps == 0:
                    print(
                        f"[mwer] step={global_step}/{total_steps} "
                        f"samples={samples_seen}/{max_samples} "
                        f"loss={batch_loss_value:.4f} "
                        f"L_mwer={mwer_value:.4f}  L_ce={ce_value:.4f}  "
                        f"elapsed={time.time()-t0:.0f}s",
                        flush=True)

                # ── Mid-training eval ──────────────────────────────────────────
                if args.eval_steps > 0 and global_step % args.eval_steps == 0 and eval_rows:
                    model.eval()
                    m = evaluate(model, processor, eval_rows, args.language,
                                 args.eval_batch_size, dtype, device, args.language)
                    tag = f"WER={m['wer']:.4f}" if m["wer"] is not None else f"CER={m['cer']:.4f}"
                    print(f"[mwer] mid-eval step={global_step}  {tag}", flush=True)
                    model.train()
                    cleanup_cuda()

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.eval()
    if eval_rows:
        print("[mwer] final eval on dev100 …")
        t_e = time.time()
        metrics = evaluate(model, processor, eval_rows, args.language,
                           args.eval_batch_size, dtype, device, args.language)
        metrics["wall_seconds"] = time.time() - t_e
        out_path = os.path.join(args.eval_out_dir, f"{args.tag}.json")
        save_eval(metrics, args, out_path)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    import shutil
    for fn in ["config.json", "generation_config.json", "preprocessor_config.json",
               "processor_config.json", "tokenizer_config.json", "tokenizer.json",
               "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json"]:
        src = os.path.join(args.model_path, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fn))
    print(f"[mwer] saved → {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR MWER fine-tuning")
    p.add_argument("--model_path",    default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--train_file",    required=True)
    p.add_argument("--eval_file",     default="")
    p.add_argument("--output_dir",    default="./qwen3-mwer-out")
    p.add_argument("--tag",           default="qwen3_mwer")
    p.add_argument("--language",      default="French",
                   choices=["French", "Chinese"])
    p.add_argument("--n_best",        type=int,   default=4)
    p.add_argument("--mwer_batch_size", type=int, default=4,
                   help="Number of audios per MWER generation/scoring microbatch")
    p.add_argument("--generation_strategy", choices=["sample", "beam"],
                   default="sample",
                   help="N-best source: faster temperature sampling or beam search")
    p.add_argument("--temperature",   type=float, default=0.9)
    p.add_argument("--top_p",         type=float, default=0.95)
    p.add_argument("--lambda_ce",     type=float, default=0.01)
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--epochs",        type=float, default=1.0)
    p.add_argument("--grad_acc",      type=int,   default=4)
    p.add_argument("--weight_decay",  type=float, default=0.0)
    p.add_argument("--warmup_ratio",  type=float, default=0.05)
    p.add_argument("--log_steps",     type=int,   default=50)
    p.add_argument("--eval_steps",    type=int,   default=200,
                   help="Mid-training eval every N optimizer steps (0=off)")
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_out_dir",
                   default="/data/speech2text/Qwen3-ASR/finetuning/outputs")
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    train_mwer(parse_args())
