# coding=utf-8
"""
GSPO (Group Sequence Policy Optimisation) fine-tuning for Qwen3-ASR.

Algorithm (Qwen team, 2025):
  1. Sample G rollouts from a stale old-policy snapshot (no grad).
  2. Score rollouts under CURRENT policy (grad flows) and OLD policy (no grad).
  3. Compute composite reward per rollout: -WER/CER  +  alpha * format_bonus.
  4. Group-relative advantage: z-score within the G rollouts per prompt.
     Skip steps where rewards have no variance (all rollouts identical).
  5. SEQUENCE-LEVEL length-normalised importance ratio:
       r_i = exp( (logp_cur_i - logp_old_i) / |y_i| )
     Length normalisation keeps ratio near 1.0 for long sequences.
  6. Asymmetric clipped surrogate (GSPO core):
       L = -min( r_i * A_i,  clip(r_i, 1-EPS_LOW, 1+EPS_HIGH) * A_i )
  7. Sync old_model <- model every SYNC_EVERY grad steps.

Clip window is tight because the sequence-level geometric-mean ratio
is close to 1.0 even after several gradient steps:
  EPS_LOW  = 3e-4
  EPS_HIGH = 4e-4

Key correctness choices:
  - padding_side="right" for scoring so prefix_lens are not displaced.
  - Rollouts scored as raw decoded text (lang tag included), WER on parsed text.
  - KL penalty omitted (GSPO paper uses none; dead-code branch removed).
"""
import argparse
import contextlib
import copy
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

# ── GSPO clip constants ───────────────────────────────────────────────────────
EPS_LOW   = 3e-4   # lower asymmetric clip on sequence-level ratio
EPS_HIGH  = 4e-4   # upper asymmetric clip
SYNC_EVERY = 32    # grad steps between old_model <- model sync

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


def has_expected_format(raw_text: str, language: str) -> bool:
    return f"language {language}<asr_text>" in raw_text


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


def compute_sequence_logp(model, inputs: dict, prefix_lens: list,
                           pad_id: int, no_grad: bool = False) -> torch.Tensor:
    """
    Return summed per-token log-probs for hypothesis tokens (vectorised).

    Requires RIGHT-padded inputs so that position plen is the first hypothesis
    token. Set processor.tokenizer.padding_side='right' before calling.

    Args:
        no_grad : True when scoring old/ref policy (saves memory)
    Returns:
        [N] float tensor
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        fwd  = {k: inputs[k] for k in _SCORE_KEYS if k in inputs}
        out  = model(**fwd)
        logits    = out.logits                     # [N, T, V]
        N, T, _   = logits.shape
        input_ids = inputs["input_ids"]            # [N, T]

        # shift: logits[n,t] predicts input_ids[n,t+1]
        shifted_ids  = input_ids[:, 1:].clamp(min=0)
        shifted_logp = F.log_softmax(logits[:, :-1].float(), dim=-1)  # fp32 for stability
        gathered     = shifted_logp.gather(2, shifted_ids.unsqueeze(-1)).squeeze(-1)  # [N,T-1]

        plen_t    = torch.tensor(prefix_lens, device=input_ids.device).unsqueeze(1)
        positions = torch.arange(T - 1, device=input_ids.device).unsqueeze(0)
        hyp_mask  = (positions >= (plen_t - 1)) & (input_ids[:, 1:] != pad_id)

        logp_sums = (gathered * hyp_mask.float()).sum(dim=1)  # [N]
    return logp_sums


def count_hyp_tokens(inputs: dict, prefix_lens: list, pad_id: int) -> torch.Tensor:
    """Non-pad hypothesis token count per sequence, for length normalisation."""
    input_ids = inputs["input_ids"]
    T = input_ids.shape[1]
    plen_t    = torch.tensor(prefix_lens, device=input_ids.device).unsqueeze(1)
    positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
    hyp_mask  = (positions >= plen_t) & (input_ids != pad_id)
    return hyp_mask.float().sum(dim=1).clamp(min=1)   # [N]


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

    orig_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    preds, refs = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        audios  = [librosa.load(r["audio"], sr=16000, mono=True)[0] for r in batch]
        inputs  = processor(text=[prompt_with_lang] * len(batch), audio=audios,
                            return_tensors="pt", padding=True)
        inputs  = cast_inputs(inputs, dtype, device)
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=processor.tokenizer.eos_token_id)
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

def train_gspo(args):
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

    # Old-policy snapshot — kept frozen for rollout generation + scoring
    old_model = copy.deepcopy(model)
    old_model.to(device).eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    pad_id  = processor.tokenizer.pad_token_id or 0
    eos_str = processor.tokenizer.eos_token or ""

    train_rows = load_jsonl(args.train_file)
    eval_rows  = load_jsonl(args.eval_file) if args.eval_file else []
    prefix_text = build_prefix_text(processor)
    lang_tag    = f"language {args.language}<asr_text>"

    max_samples  = int(len(train_rows) * args.epochs)
    trainable    = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    accum_target = max(args.grad_acc, args.gspo_batch_size)
    total_steps  = math.ceil(max_samples / accum_target)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)

    global_step  = 0
    samples_seen = 0
    accum_samples = 0
    skipped      = 0
    optimizer.zero_grad()
    t0 = time.time()

    while samples_seen < max_samples:
        random.shuffle(train_rows)
        for start in range(0, len(train_rows), args.gspo_batch_size):
            if samples_seen >= max_samples:
                break
            batch = train_rows[start:start + args.gspo_batch_size]
            remaining = max_samples - samples_seen
            batch = batch[:remaining]
            if not batch:
                break
            batch_size = len(batch)
            samples_seen += batch_size

            audios = [librosa.load(r["audio"], sr=16000, mono=True)[0] for r in batch]
            audio_cache = build_audio_cache(processor, audios)

            # ── 1. Sample G rollouts from OLD policy (left-pad for generation) ──
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

            with generation_cache_enabled(old_model):
                with torch.no_grad():
                    gen_out = old_model.generate(
                        **gen_inputs,
                        max_new_tokens=256,
                        num_return_sequences=args.group_size,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out

            raw_hyps = []
            clean_hyps = []
            for b in range(batch_size):
                raw_row, clean_row = [], []
                for g in range(args.group_size):
                    seq_i = b * args.group_size + g
                    cont_ids = sequences[seq_i, prompt_width:]
                    raw = processor.decode(
                        cont_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                    _, parsed = parse_asr_output(raw, user_language=None)
                    raw_row.append(raw)
                    clean_row.append(parsed)
                raw_hyps.append(raw_row)
                clean_hyps.append(clean_row)

            # ── 2. Composite reward: -WER/CER  +  format bonus ────────────────
            rewards_rows = []
            for row, raw_row, clean_row in zip(batch, raw_hyps, clean_hyps):
                rewards_list = []
                for raw_text, clean_text in zip(raw_row, clean_row):
                    err    = compute_error_rate(clean_text, row["reference"], args.language)
                    fmt_ok = float(has_expected_format(raw_text, args.language))
                    rewards_list.append(-err + args.format_alpha * fmt_ok)
                rewards_rows.append(rewards_list)
            rewards = torch.tensor(rewards_rows, device=device, dtype=torch.float32)  # [B,G]

            # ── 3. Skip degenerate steps (all rollouts identical) ──────────────
            active_mask = rewards.std(dim=1) >= 1e-3
            if not bool(active_mask.any()):
                skipped += batch_size
                continue
            active_indices = [i for i, keep in enumerate(active_mask.tolist()) if keep]
            skipped += batch_size - len(active_indices)
            active_count = len(active_indices)

            # ── 4. Group-relative advantage (z-score) ─────────────────────────
            rewards_active = rewards[active_mask]  # [B_active,G]
            mu    = rewards_active.mean(dim=1, keepdim=True)
            sigma = rewards_active.std(dim=1, keepdim=True)
            A     = (rewards_active - mu) / sigma.clamp(min=1e-6)    # [B_active,G]

            # ── 5. Build scoring inputs (right-pad for correct offset) ─────────
            processor.tokenizer.padding_side = "right"
            # Score raw rollout text (contains lang tag already)
            flat_audio_indices = [b for b in active_indices for _ in range(args.group_size)]
            flat_raw_hyps = [h for b in active_indices for h in raw_hyps[b]]
            full_texts = [prefix_text + h + eos_str for h in flat_raw_hyps]

            full_inputs = processor_with_cached_audio(
                processor,
                texts=full_texts,
                audio_cache=audio_cache,
                audio_indices=flat_audio_indices,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            prefix_inputs_1 = processor_with_cached_audio(
                processor,
                texts=[prefix_text] * active_count,
                audio_cache=audio_cache,
                audio_indices=active_indices,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            full_inputs     = cast_inputs(full_inputs,     dtype, device)
            prefix_inputs_1 = cast_inputs(prefix_inputs_1, dtype, device)
            plen_per_sample = prefix_inputs_1["attention_mask"].sum(dim=1).tolist()
            prefix_lens = [plen_per_sample[i]
                           for i in range(active_count)
                           for _ in range(args.group_size)]

            # ── 6. Score under CURRENT policy (grad) and OLD policy (no grad) ──
            logp_cur = compute_sequence_logp(model,     full_inputs, prefix_lens, pad_id,
                                              no_grad=False).view(active_count, args.group_size)
            logp_old = compute_sequence_logp(old_model, full_inputs, prefix_lens, pad_id,
                                              no_grad=True).detach().view(active_count, args.group_size)

            # ── 7. Sequence-level length-normalised ratio ──────────────────────
            lengths   = count_hyp_tokens(full_inputs, prefix_lens, pad_id).view(active_count, args.group_size)
            log_ratio = (logp_cur - logp_old) / lengths
            ratio     = torch.exp(log_ratio)    # [B_active,G]

            # ── 8. Asymmetric clipped surrogate ───────────────────────────────
            unclipped = ratio * A
            clipped   = torch.clamp(ratio, 1.0 - EPS_LOW, 1.0 + EPS_HIGH) * A
            L_policy  = -torch.minimum(unclipped, clipped).mean()

            loss = L_policy * (active_count / accum_target)
            loss.backward()
            accum_samples += active_count

            # ── 9. Optimiser step ──────────────────────────────────────────────
            if accum_samples >= accum_target or samples_seen == max_samples:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_samples = 0

                # Sync old policy
                if global_step % SYNC_EVERY == 0:
                    old_model.load_state_dict(model.state_dict())
                    old_model.eval()
                    for p in old_model.parameters():
                        p.requires_grad_(False)

                if global_step % args.log_steps == 0:
                    print(
                        f"[gspo] step={global_step}/{total_steps} "
                        f"samples={samples_seen}/{max_samples} "
                        f"loss={L_policy.item():.4f} "
                        f"mean_r={rewards_active.mean().item():.4f}  "
                        f"ratio_mean={ratio.mean().item():.6f}  "
                        f"skipped={skipped}  elapsed={time.time()-t0:.0f}s",
                        flush=True)

                if args.eval_steps > 0 and global_step % args.eval_steps == 0 and eval_rows:
                    model.eval()
                    m = evaluate(model, processor, eval_rows, args.language,
                                 args.eval_batch_size, dtype, device, args.language)
                    tag = f"WER={m['wer']:.4f}" if m["wer"] is not None else f"CER={m['cer']:.4f}"
                    print(f"[gspo] mid-eval step={global_step}  {tag}", flush=True)
                    model.train()

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.eval()
    if eval_rows:
        print("[gspo] final eval on dev100 …")
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
    print(f"[gspo] saved → {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR GSPO fine-tuning")
    p.add_argument("--model_path",    default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--train_file",    required=True)
    p.add_argument("--eval_file",     default="")
    p.add_argument("--output_dir",    default="./qwen3-gspo-out")
    p.add_argument("--tag",           default="qwen3_gspo")
    p.add_argument("--language",      default="French",
                   choices=["French", "Chinese"])
    p.add_argument("--group_size",    type=int,   default=4)
    p.add_argument("--gspo_batch_size", type=int, default=2,
                   help="Number of audios per GSPO rollout/scoring microbatch")
    p.add_argument("--temperature",   type=float, default=0.7,
                   help="Sampling temperature for rollout generation")
    p.add_argument("--format_alpha",  type=float, default=0.1,
                   help="Format-compliance bonus weight in reward")
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--epochs",        type=float, default=1.0)
    p.add_argument("--grad_acc",      type=int,   default=4)
    p.add_argument("--weight_decay",  type=float, default=0.0)
    p.add_argument("--warmup_ratio",  type=float, default=0.05)
    p.add_argument("--log_steps",     type=int,   default=50)
    p.add_argument("--eval_steps",    type=int,   default=200)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_out_dir",
                   default="/data/speech2text/Qwen3-ASR/finetuning/outputs")
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    train_gspo(parse_args())
