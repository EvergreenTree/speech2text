"""Prepare 500-utterance test slices for FLEURS fr_fr and CV21 zh-CN.

Mirrors the existing repo's selection (filter clip duration <= 30s, text >= 2 chars,
shuffle seed=42, take 500). Saves audio arrays + refs to a fast-loading pickle so
each model run doesn't redo audio decoding / resampling.
"""
from __future__ import annotations

import os
import pickle
import sys

os.environ.setdefault("HF_HOME", "/data/speech2text/outputs/cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/speech2text/outputs/cache")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

from datasets import load_dataset, Audio

OUT_DIR = "/data/speech2text/asr_bench"
SAMPLING_RATE = 16_000
MAX_AUDIO_SEC = 30.0
MIN_TEXT_CHARS = 2
SEED = 42
N_TEST = 500


def filter_clip(text_field: str):
    def _flt(ex) -> bool:
        sentence = ex.get(text_field) or ""
        if len(sentence.strip()) < MIN_TEXT_CHARS:
            return False
        au = ex["audio"]
        if len(au["array"]) / au["sampling_rate"] > MAX_AUDIO_SEC:
            return False
        return True
    return _flt


def prepare(profile_name: str, dataset_id: str, config: str | None,
            text_field: str, trust_remote_code: bool):
    print(f"\n=== preparing {profile_name} ===")
    if config is not None:
        ds = load_dataset(dataset_id, config, split="test",
                          cache_dir=os.environ["HF_DATASETS_CACHE"],
                          trust_remote_code=trust_remote_code)
    else:
        ds = load_dataset(dataset_id, split="test",
                          cache_dir=os.environ["HF_DATASETS_CACHE"],
                          trust_remote_code=trust_remote_code)
    print(f"raw test n = {len(ds)}")

    # Cast to 16 kHz once.
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.filter(filter_clip(text_field))
    print(f"after filter n = {len(ds)}")

    # Match repo: shuffle(seed=42), select(range(500)).
    ds = ds.shuffle(seed=SEED).select(range(min(N_TEST, len(ds))))
    print(f"final n = {len(ds)}")

    items = []
    for ex in ds:
        items.append({
            "audio": ex["audio"]["array"].astype("float32"),
            "ref": ex[text_field],
            "sr": ex["audio"]["sampling_rate"],
        })

    out_path = os.path.join(OUT_DIR, f"test_{profile_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(items, f)
    total_sec = sum(len(it["audio"]) / it["sr"] for it in items)
    print(f"saved {out_path}: {len(items)} clips, {total_sec / 60:.1f} min audio")
    return out_path


def main():
    prepare("fleurs_fr", "google/fleurs", "fr_fr", "transcription", True)
    prepare("cv21_zh", "keeve101/common-voice-21.0-2025-03-14-zh-CN-split",
            None, "sentence", False)


if __name__ == "__main__":
    main()
