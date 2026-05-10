#!/usr/bin/env bash
# Phase D — paradigmes alternatifs / références externes (zero-shot).
# Sert à situer nos fine-tunes maison dans le paysage français existant.
set -euo pipefail
VENV=/data/venv
PY="$VENV/bin/python"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HOME=/data/speech2text/outputs/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
ROOT=/data/speech2text
cd "$ROOT"

# Référence 1 : wav2vec2 / XLS-R 1B fine-tuné sur Common Voice 9 fr.
# Paradigme alternatif : encodeur SSL + tête CTC, pas de seq2seq.
# Sur le même test FLEURS fr_fr, en zero-shot.
echo "=== ref_w2v_fr (zero-shot eval) ==="
$PY -m src.eval_w2v --model bofenghuang/asr-wav2vec2-ctc-french \
    --tag ref_w2v_fr --split test --n 500 --batch-size 4 --dtype fp16 \
    2>&1 | tee outputs/logs/eval_ref_w2v_fr.log

# Référence 2 : Whisper-large-v3 distillé en français (decoder 4 couches),
# publié par bofenghuang. Modèle français spécialisé, même paradigme seq2seq
# que nos runs, zero-shot — pour mesurer combien notre LoRA-turbo « rattrape »
# par rapport à un modèle déjà tuné en français.
echo "=== ref_whisper_fr_distil (zero-shot eval) ==="
RAW_T="outputs/cache/processed/openai__whisper-large-v3-turbo/raw"
if [[ -d "$RAW_T" ]]; then
    $PY -m src.eval --model bofenghuang/whisper-large-v3-french-distil-dec4 \
        --raw-dir "$RAW_T" --split test --tag ref_whisper_fr_distil \
        --batch-size 4 \
        2>&1 | tee outputs/logs/eval_ref_whisper_fr_distil.log
else
    echo "skipping ref_whisper_fr_distil — no turbo raw dir yet"
fi

echo "PHASE D DONE"
