"""Data loading + preprocessing for Whisper fine-tuning on FLEURS fr_fr.

Loads `google/fleurs` config `fr_fr` (a few hours of read speech, 16 kHz mono,
already split into train / validation / test). FLEURS is the standard
multilingual ASR benchmark — published Whisper / SeamlessM4T / Parakeet
numbers are directly comparable, which is what we want for the report.

Two transcript fields are available:
- `transcription`     : lowercased, punctuation stripped (the FLEURS-standard target).
- `raw_transcription` : true-cased, with punctuation.

We train and evaluate on `transcription` so the metric is comparable across runs
without needing a separate normalization stage. We also keep `raw_transcription`
in the raw split for inspection.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

from datasets import Audio, DatasetDict, load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

DATASET_ID = "google/fleurs"
DATASET_CONFIG = "fr_fr"
SAMPLING_RATE = 16_000
MAX_AUDIO_SEC = 30.0
MIN_TEXT_CHARS = 2

LANGUAGE = "fr"
TASK = "transcribe"

# Dataset profiles — text field name and split mapping vary across mirrors.
DATASET_PROFILES = {
    "fleurs-fr": {
        "id": "google/fleurs",
        "config": "fr_fr",
        "text_field": "transcription",
        "language": "fr",
        "trust_remote_code": True,
    },
    "cv21-zh": {
        "id": "keeve101/common-voice-21.0-2025-03-14-zh-CN-split",
        "config": None,
        "text_field": "sentence",
        "language": "zh",
        "trust_remote_code": False,
    },
}


def load_splits(profile: str | None = None, dataset_id: str | None = None,
                config: str | None = None, cache_dir: str | None = None,
                trust_remote_code: bool = True):
    if profile is not None:
        p = DATASET_PROFILES[profile]
        dataset_id = p["id"]
        config = p["config"]
        trust_remote_code = p["trust_remote_code"]
    if config is not None:
        return load_dataset(dataset_id, config, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    return load_dataset(dataset_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)


def deterministic_subsample(ds, n: int | None, seed: int = 42):
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def filter_clip_factory(text_field: str = "transcription"):
    def _filter(ex) -> bool:
        sentence = ex.get(text_field) or ex.get("transcription") or ex.get("sentence") or ""
        if len(sentence.strip()) < MIN_TEXT_CHARS:
            return False
        audio = ex["audio"]
        if len(audio["array"]) / audio["sampling_rate"] > MAX_AUDIO_SEC:
            return False
        return True
    return _filter


def filter_clip(ex) -> bool:
    return filter_clip_factory("transcription")(ex)


def make_processor(model_id: str, language: str = LANGUAGE):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language=language, task=TASK)
    return feature_extractor, tokenizer


def encode_factory(feature_extractor, tokenizer, text_field: str = "transcription"):
    def _encode(ex):
        audio = ex["audio"]
        ex["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        ex["labels"] = tokenizer(ex[text_field]).input_ids
        return ex

    return _encode


def prepare_splits(
    model_id: str,
    n_train: int | None = None,
    n_dev: int | None = None,
    n_test: int | None = 500,
    cache_dir: str | None = None,
    num_proc: int = 4,
    seed: int = 42,
    language: str = LANGUAGE,
    profile: str = "fleurs-fr",
    text_field: str | None = None,
):
    """Load the chosen profile, optionally subsample, return raw + processed DatasetDict.

    `raw` keeps the audio for inference / inspection.
    `processed` has only `input_features` and `labels` for fast training.

    Audio is cast to 16 kHz mono regardless of source.
    """
    p = DATASET_PROFILES[profile]
    if text_field is None:
        text_field = p["text_field"]
    raw = load_splits(profile=profile, cache_dir=cache_dir)
    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Standardize split names.
    if "validation" in raw and "dev" not in raw:
        raw["dev"] = raw["validation"]

    flt = filter_clip_factory(text_field)
    raw["train"] = raw["train"].filter(flt, num_proc=num_proc)
    raw["dev"] = raw["dev"].filter(flt, num_proc=num_proc)
    raw["test"] = raw["test"].filter(flt, num_proc=num_proc)

    subsampled = DatasetDict(
        train=deterministic_subsample(raw["train"], n_train, seed=seed),
        dev=deterministic_subsample(raw["dev"], n_dev, seed=seed),
        test=deterministic_subsample(raw["test"], n_test, seed=seed),
    )

    feature_extractor, tokenizer = make_processor(model_id, language=language)
    encode_fn = encode_factory(feature_extractor, tokenizer, text_field=text_field)

    processed = DatasetDict()
    for split_name, ds in subsampled.items():
        processed[split_name] = ds.map(
            encode_fn,
            remove_columns=ds.column_names,
            num_proc=num_proc,
            desc=f"encode {split_name}",
        )
    return subsampled, processed, (feature_extractor, tokenizer)


@dataclass
class WhisperDataCollator:
    """Pad audio features and labels independently (Whisper's standard recipe)."""

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="openai/whisper-small")
    ap.add_argument("--n-train", type=int, default=None,
                    help="cap on number of train examples; default = all (~3193 in FLEURS fr_fr)")
    ap.add_argument("--n-dev", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--cache-dir", default="/data/speech2text/outputs/cache/datasets")
    ap.add_argument("--out-dir", default="/data/speech2text/outputs/cache/processed")
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--language", default="fr")
    ap.add_argument("--profile", default="fleurs-fr",
                    choices=sorted(DATASET_PROFILES.keys()),
                    help="Dataset profile (fleurs-fr | cv21-zh)")
    args = ap.parse_args()

    raw, processed, _ = prepare_splits(
        model_id=args.model_id,
        n_train=args.n_train,
        n_dev=args.n_dev,
        n_test=args.n_test,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
        language=args.language,
        profile=args.profile,
    )
    # Keep flat layout for the default fr profile (matches existing outputs/);
    # prefix with profile name for additional profiles to avoid collisions.
    if args.profile == "fleurs-fr":
        out_root = os.path.join(args.out_dir, args.model_id.replace("/", "__"))
    else:
        out_root = os.path.join(args.out_dir, args.profile, args.model_id.replace("/", "__"))
    os.makedirs(out_root, exist_ok=True)
    processed.save_to_disk(os.path.join(out_root, "processed"))
    raw.save_to_disk(os.path.join(out_root, "raw"))
    print(f"saved processed: {os.path.join(out_root, 'processed')}")
    print(f"saved raw:       {os.path.join(out_root, 'raw')}")
    print({k: len(v) for k, v in processed.items()})


if __name__ == "__main__":
    main()
