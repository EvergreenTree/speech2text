# Whisper zh-CN — fine-tuning under compute constraints

## TL;DR

Fine-tuned `openai/whisper-small` on **Common Voice 21 (zh-CN)** with LoRA and a small full-FT
control, then scaled to `openai/whisper-medium` with LoRA. **Best result: CER 13.22 % on
medium + LoRA, a 54 % relative reduction over the medium baseline (28.73 %).**
Single L4 GPU (23 GB VRAM, well under 16 GB at runtime), bf16 throughout, ~57 min total
training time across the three runs (well inside the 4 GPU-hour budget).

A live Gradio server is included so you can record / upload audio in your browser and compare
baseline and fine-tuned outputs side-by-side.

## Repo layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data.py        # CV21 zh-CN load → resample → filter → encode features/labels
│   ├── train.py       # CLI: --mode lora | full
│   ├── eval.py        # generation + CER/MER on test split
│   ├── analyze.py     # 5-row table + worst-100 error bucketing
│   └── server.py      # Gradio app (mic + upload, baseline vs fine-tuned)
├── scripts/run_all.sh # end-to-end pipeline
└── outputs/
    ├── preds/         # JSON: predictions + per-utterance refs/hyps
    ├── adapters/      # LoRA weights / full-FT checkpoints
    ├── logs/          # stdout from each step
    ├── metrics.json   # final 5-row results table
    └── error_analysis.json
```

## 1. Dataset

We use **Common Voice 21 (zh-CN)** via the parquet-formatted re-host
[`keeve101/common-voice-21.0-2025-03-14-zh-CN-split`](https://huggingface.co/datasets/keeve101/common-voice-21.0-2025-03-14-zh-CN-split).
The Mozilla-hosted `mozilla-foundation/common_voice_17_0` repo is gated behind a click-through
license and ships its data via a loader script that the current `datasets` 4.x runtime no longer
executes; FLEURS has the same `datasets`-script issue.  The keeve101 mirror exposes plain
parquet shards at 32 kHz mp3, with clean `train/dev/test` splits already partitioned.

**Characteristics:** read speech (volunteer-recorded), single speaker per clip, ~3-7 s utterances,
sentences from Wikipedia and other open text. Heavy long-tail vocabulary: place names, person names,
historical text. Total dataset is ~2 GB (29k train / 10k dev / 11k test before subsampling). After
filtering (≤30 s clip, ≥2-char transcript) and a deterministic shuffle (seed=42) we use **4000 train /
300 dev / 500 test** to fit the 4-GPU-hour budget.

**Preprocessing:**
1. Cast `audio` column to 16 kHz mono via `Audio(sampling_rate=16_000)` (CV21 mp3 is 32 kHz).
2. `WhisperFeatureExtractor` → 80-bin log-mel × 3000 frames per clip (Whisper's standard input).
3. `WhisperTokenizer(language="zh", task="transcribe")` → token IDs with the SOT/zh/transcribe prefix.
4. The `WhisperDataCollator` pads features and labels independently and replaces label-pad with `-100`.

## 2. Model & training strategy

We bootstrap on **Whisper-small** (244 M params) then scale to **Whisper-medium** (769 M).

**Why LoRA?** Whisper is multi-task multilingual: the pre-trained representations are valuable and
full FT on a few thousand utterances overfits quickly while throwing away that prior. LoRA on the
attention projections (q/k/v/out_proj, encoder + decoder) gives us ~7 M trainable params on small
(2.8 % of base) and shifts the output distribution toward simplified-Chinese CV-style transcription
without rewriting the acoustic encoder. With the L4's 23 GB VRAM this also lets us keep effective
batch size healthy on medium without aggressive gradient checkpointing.

**Why also a small full-FT control?** To verify the bootstrap pipeline end-to-end and confirm that
LoRA is not leaving large gains on the table at this scale. We run it only at the small size with a
conservative LR (1e-5), per the spec.

**Hyper-parameters (final):**

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Precision | bf16 (L4 supports it) |
| LR (LoRA) | 1e-4 |
| LR (full FT) | 1e-5 |
| Warmup ratio | 0.1 |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| LoRA targets | `q_proj, k_proj, v_proj, out_proj` (encoder + decoder) |
| Effective batch (small) | 16 (per-device 16, accum 1) |
| Effective batch (medium) | 16 (per-device 8, accum 2, grad ckpt) |
| Epochs (small) | 1 |
| Epochs (medium) | 2 |
| Seed | 42 |
| Generation | greedy, max_new_tokens=225, forced lang=zh task=transcribe |

## 3. Results

All 5 rows are evaluated on the **same 500-utterance test slice** (deterministic seed=42 from
CV21 zh-CN test split). Greedy decoding, bf16, num_beams=1.

| Run | Trainable params | Train wall | Test CER | Test MER | Δ CER (abs) | Δ CER (rel) |
|---|---:|---:|---:|---:|---:|---:|
| Whisper-small **baseline**     | —          | —      | **0.3352** | 0.3327 | —        | —        |
| Whisper-small + **LoRA**       | 7.1 M      | 8.9 m  | 0.2322     | 0.2206 | −0.1030 | −30.7 % |
| Whisper-small **full FT**      | 244 M      | 8.9 m  | 0.2208     | 0.2100 | −0.1144 | −34.1 % |
| Whisper-medium **baseline**    | —          | —      | **0.2873** | 0.2821 | —        | —        |
| Whisper-medium + **LoRA**      | 18.9 M     | 39.9 m | **0.1322** | **0.1318** | **−0.1551** | **−54.0 %** |

Numbers come from `outputs/metrics.json`; raw predictions are in `outputs/preds/`.

Reads:
- LoRA on whisper-small recovers ~30 % of the CER. Full FT at the same scale is only marginally
  better (1.1 abs CER pt). At ~3 % trainable params, LoRA is the better return on engineering
  budget here.
- Whisper-medium baseline (28.73 %) is *worse than whisper-small + LoRA* (23.22 %). A larger
  base model is not a substitute for a small amount of in-domain fine-tuning.
- Whisper-medium + LoRA halves the CER relative to its baseline. The two-epoch / larger-model
  combination is what unlocks the sub-15 % regime.
- A simple **95 % Wilson confidence interval** for medium-LoRA CER on 500 clips × ~13 chars
  per clip ≈ 6500 chars → ±0.8 abs CER pt. The 15.5 abs CER pt gap to baseline is ≫ that.

## 4. Error analysis

Worst 100 utterances from `lora_medium` were bucketed (see `outputs/error_analysis.json`).
Categories (an utterance can fall into more than one):

| Category | Count |
|---|---:|
| character_substitution | 108 |
| other | 39 |
| insertion | 5 |
| deletion | 5 |
| character_homophone | 2 |
| truncation | 2 |
| hallucination_long | 1 |
| latin_inserted | 1 |

Qualitative themes:
- **Proper-noun substitutions** dominate. e.g. `保山线` → `宝山县` (a railway line
  → a county); the audio has `bǎo shān xiàn` and the model picks the more frequent
  homophone. CV21 transcripts come from Wikipedia, so the long tail of place / person names
  is irreducibly hard with 4000 train utterances.
- **Repeated fallback.** When the input is dense with rare characters, the medium-LoRA model
  occasionally collapses to a generic-looking `他在中国的大学中学生` phrase — visible across
  several otherwise unrelated truncation / insertion errors. Symptom of LoRA pulling the
  decoder toward training-set bigrams when the audio prior is weak.
- **他/她 homophones.** Same pinyin (`tā`) for he/she/it. CER counts these as substitutions
  even though they are acoustically indistinguishable; could be normalized away in a
  downstream metric if the application allows.
- **Simplified-vs-traditional drift** seen in the baseline (e.g. `菱形是四边相等` →
  `連顯示四邊相當`) is almost entirely fixed by fine-tuning, since CV21 is simplified.
- **Latin insertions** are rare after fine-tuning (1 of 100). Baseline had more (`N性洞窖`,
  numerals as Arabic).

## 5. Limitations & trade-offs

- **CV17 was unavailable to the runtime.** The Mozilla CV repos require their loader script,
  which the new `datasets` 4.x runtime no longer executes. We pinned `datasets<4` and used the
  parquet-formatted CV21 zh-CN community re-host
  (`keeve101/common-voice-21.0-2025-03-14-zh-CN-split`). CV21 is roughly comparable to CV17 in
  zh-CN size; the field of comparison is consistent because all five table rows use the
  *same* 500-utterance test split.
- **Single-epoch training on small.** Picked to fit the 4 GPU-hour budget across five runs and
  preserve headroom for a medium-LoRA run. With more compute we would early-stop on dev CER.
- **Greedy decoding.** Beam search would likely shave a fraction of a CER point but multiplies
  wall-clock per clip. The Gradio server exposes a `num_beams` slider so you can verify
  on your own samples.
- **No augmentation.** SpecAugment would help for the rare-word long tail (which is the main
  remaining error mode) but adds engineering surface inside the time budget.
- **MER on Chinese.** jiwer's word-level MER is degenerate without whitespace; we space-join
  characters before computing it, making it close to but not identical to CER.
- **CV21 transcripts are Wikipedia-style.** Heavy on place / person names. Fine-tuning will
  not give the same gains on conversational or read-news data without more diverse training
  sources.
- **No dev-driven early stopping.** With `predict_with_generate=True` HF Trainer would have
  evaluated CER each `eval_strategy="steps"` interval, but generation is slow enough on
  whisper-medium that it would have eaten our budget. We fixed step counts up front and
  trusted the small-scale bootstrap to validate the pipeline.

## 6. Transfer to Arabic

The same scaffolding transfers, but the failure modes shift:
- **Dialect splits.** MSA vs Egyptian/Levantine/Maghrebi diverge enough that one fine-tune does not
  cover them well. We would either pick the dialect closest to the deployment domain or train a
  multi-dialect adapter and condition on a dialect tag.
- **Diacritics.** Most production text is unvocalized; metrics should be computed with diacritics
  stripped (or both ways reported) to avoid penalizing reasonable hypotheses.
- **Hamza / alef normalization.** `أ ا إ آ` collapse to `ا`, `ى → ي`, `ة → ه` are standard
  pre-eval normalization steps.
- **Code-switching.** English insertions in Arabic recordings are common; the LoRA recipe should
  keep encoder weights free enough to handle bilingual frames (we already train both encoder and
  decoder LoRA).
- **Tokenizer.** WER over orthographic words is more meaningful in Arabic than in Chinese, so we
  would lead with WER and report CER as a secondary metric.

What stays the same: dataset filtering, the LoRA recipe (rank 32, alpha 64, q/k/v/out_proj),
bf16, greedy decoding, and the Gradio harness.

## 7. With more time

- Real `dev`-driven early stopping with `predict_with_generate=True` (slow but principled).
- LoRA rank sweep `{8, 16, 32, 64}` × LR `{5e-5, 1e-4, 3e-4}`.
- SpecAugment + audio augmentation (room IR, additive noise) for the long tail.
- Whisper-large-v3 LoRA (would need 8-bit base or gradient checkpointing on L4).
- LM rescoring via a small zh n-gram model on Wikipedia for the rare-word substitutions.

## 8. Reproduction

Pinned: `transformers==5.6.0`, `datasets>=3.6,<4.0`, `peft==0.19.1`, `accelerate==1.13.0`,
`torch==2.11.0+cu128`, `jiwer==4.0.0`, `librosa==0.11.0`, `gradio==6.14.0`. Seed=42.

```bash
# Install (reuse the existing /data/venv on the test machine)
/data/venv/bin/pip install -r requirements.txt

# Authenticated Hugging Face access (CV21 mirror is open, but HF_TOKEN avoids rate limits)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache

# End-to-end (data prep + 5-row table + analysis)
bash scripts/run_all.sh

# Live demo: mic / upload + baseline-vs-fine-tuned side-by-side
bash scripts/start_demo.sh
# overrides via env vars: BASE, LORA, FULL, SERVER_PORT, SERVER_HOST
```

Defaults to `whisper-medium + lora_medium` if both exist, else `whisper-small + lora_small`.

**Accessing the demo from your laptop (recommended):** SSH-tunnel the port and open in
your local browser — this is the way that gives you working microphone capture, because
browsers require a secure context (HTTPS *or* `localhost`) for `getUserMedia`:

```bash
# from your laptop
ssh -L 7860:localhost:7860 user@<server-ip>
# then open http://localhost:7860
```

Direct `http://<server-ip>:7860` access also works for **file upload** but the browser will
silently block microphone capture because it's not localhost / HTTPS. Pass `--share` to
`src.server` for a `*.gradio.live` HTTPS link if you do not want to tunnel.

**System dependency:** the Gradio audio component decodes uploaded files through `ffmpeg`.
On a fresh machine: `sudo apt-get install -y ffmpeg`.
