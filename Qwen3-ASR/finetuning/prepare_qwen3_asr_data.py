import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio

REPO_ROOT = Path(__file__).resolve().parents[2]
WHISPER_SRC = REPO_ROOT / "whisper" / "src"
if str(WHISPER_SRC) not in sys.path:
    sys.path.insert(0, str(WHISPER_SRC))

from data import DATASET_PROFILES, deterministic_subsample, filter_clip_factory, load_splits

SAMPLING_RATE = 16_000

PROFILE_DEFAULTS = {
    "fleurs-fr": {
        "language": "French",
        "text_field": "transcription",
        "n_train": 3193,
        "n_dev": 289,
        "n_test": 500,
    },
    "cv21-zh": {
        "language": "Chinese",
        "text_field": "sentence",
        "n_train": 4000,
        "n_dev": 300,
        "n_test": 500,
    },
}


def export_split(ds, split_name: str, out_dir: Path, text_field: str, language: str):
    split_dir = out_dir / split_name
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = split_dir / f"{split_name}.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for idx, ex in enumerate(ds):
            audio = ex["audio"]
            wav_path = audio_dir / f"{idx:05d}.wav"
            sf.write(wav_path, audio["array"], audio["sampling_rate"])
            transcript = (ex.get(text_field) or "").strip()
            row = {
                "audio": str(wav_path),
                "text": f"language {language}<asr_text>{transcript}",
                "prompt": "",
                "reference": transcript,
                "language": language,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return jsonl_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=sorted(PROFILE_DEFAULTS))
    ap.add_argument("--out-dir", default="/data/speech2text/Qwen3-ASR/finetuning/data")
    ap.add_argument("--cache-dir", default="/data/speech2text/outputs/cache/datasets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-proc", type=int, default=4)
    args = ap.parse_args()

    defaults = PROFILE_DEFAULTS[args.profile]
    dataset_meta = DATASET_PROFILES[args.profile]
    raw = load_splits(profile=args.profile, cache_dir=args.cache_dir)
    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    if "validation" in raw and "dev" not in raw:
        raw["dev"] = raw["validation"]

    flt = filter_clip_factory(defaults["text_field"])
    raw["train"] = raw["train"].filter(flt, num_proc=args.num_proc)
    raw["dev"] = raw["dev"].filter(flt, num_proc=args.num_proc)
    raw["test"] = raw["test"].filter(flt, num_proc=args.num_proc)

    splits = {
        "train": deterministic_subsample(raw["train"], defaults["n_train"], seed=args.seed),
        "dev": deterministic_subsample(raw["dev"], defaults["n_dev"], seed=args.seed),
        "test": deterministic_subsample(raw["test"], defaults["n_test"], seed=args.seed),
    }

    out_root = Path(args.out_dir) / args.profile
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "profile": args.profile,
        "dataset_id": dataset_meta["id"],
        "dataset_config": dataset_meta["config"],
        "sampling_rate": SAMPLING_RATE,
        "language": defaults["language"],
        "text_field": defaults["text_field"],
        "sizes": {name: len(ds) for name, ds in splits.items()},
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for split_name, ds in splits.items():
        path = export_split(
            ds=ds,
            split_name=split_name,
            out_dir=out_root,
            text_field=defaults["text_field"],
            language=defaults["language"],
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
