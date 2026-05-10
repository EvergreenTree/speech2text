#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/data/speech2text}"
DST="${2:-/data/speech2text_export}"

mkdir -p "${DST}"

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.claude/' \
  --exclude '.gradio/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.bin' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  --exclude '*.safetensors' \
  --exclude 'checkpoint-*/' \
  --exclude 'asr_bench/cache/' \
  --exclude 'asr_bench/logs/' \
  --exclude 'asr_bench/test_*.pkl' \
  --exclude 'outputs/cache/' \
  --exclude 'outputs/logs/' \
  --exclude 'outputs/adapters/' \
  --exclude 'granite_speech/finetuning/tmp/' \
  --exclude 'granite_speech/finetuning/outputs/adapters/' \
  --exclude 'granite_speech/finetuning/outputs/logs/' \
  --exclude 'Qwen3-ASR/.git/' \
  --exclude 'Qwen3-ASR/finetuning/data/' \
  --exclude 'Qwen3-ASR/finetuning/tmp/' \
  --exclude 'Qwen3-ASR/finetuning/outputs/adapters/' \
  --exclude 'Qwen3-ASR/finetuning/outputs/logs/' \
  "${SRC}/" "${DST}/"

echo "Exported clean tree to ${DST}"
