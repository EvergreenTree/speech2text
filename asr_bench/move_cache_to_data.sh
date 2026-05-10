#!/usr/bin/env bash
# Move HF model cache from / to /data, then symlink the old path so existing
# scripts don't need to change. Run this AFTER the Whisper family run finishes.
set -e

SRC=/data/speech2text/asr_bench/cache
DST=/data/speech2text/asr_bench/cache

if [ -L "$SRC" ]; then
  echo "$SRC is already a symlink → $(readlink -f $SRC)"
  exit 0
fi

if [ ! -d "$SRC" ]; then
  echo "$SRC does not exist or is not a directory"
  exit 1
fi

mkdir -p /data/speech2text/asr_bench
echo "moving $SRC → $DST ..."
mv "$SRC" "$DST"
ln -s "$DST" "$SRC"
echo "done. $SRC is now a symlink to $DST"
ls -la "$SRC"
df -h / /data | tail -3
du -sh "$DST"
