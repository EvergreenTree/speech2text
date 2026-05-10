# ASR benchmark — fr-FR / zh-CN

### fr-FR FLEURS (500 test utterances)

| Tag | Model | Family | Params | WER | CER | Wall (s) |
|---|---|---|---:|---:|---:|---:|
| `tiny_fr` | `openai/whisper-tiny` | whisper | 38 M | 45.53% | 20.33% | 22 |
| `base_fr` | `openai/whisper-base` | whisper | 73 M | 28.62% | 12.03% | 25 |
| `small_fr` | `openai/whisper-small` | whisper | 242 M | 13.86% | 5.08% | 43 |
| `parakeet_tdt_fr` | `nvidia/parakeet-tdt-0.6b-v3` | parakeet | 627 M | 5.28% | 1.91% | 23 |
| `medium_fr` | `openai/whisper-medium` | whisper | 764 M | 8.20% | 2.92% | 223 |
| `distilfr_fr` | `bofenghuang/whisper-large-v3-french-distil-dec4` | whisper | 809 M | 6.97% | 2.56% | 66 |
| `turbo_fr` | `openai/whisper-large-v3-turbo` | whisper | 809 M | 5.79% | 2.00% | 67 |
| `largev2_fr` | `openai/whisper-large-v2` | whisper | 1.54 B | 6.30% | 2.24% | 407 |
| `largev3_fr` | `openai/whisper-large-v3` | whisper | 1.54 B | 5.73% | 1.97% | 248 |
| `canary_qwen_fr` | `nvidia/canary-qwen-2.5b` | canary | 2.56 B | 83.52% | 45.61% | 306 |
| `voxtral_mini_fr` | `mistralai/Voxtral-Mini-3B-2507` | voxtral | 4.68 B | 4.75% | 1.93% | 276 |
| `voxtral_small4bit_fr` | `mistralai/Voxtral-Small-24B-2507` | voxtral_4bit | 12.82 B | 3.97% | 1.73% | 1725 |

**Notes:**
- `canary_qwen_fr`: **English-only model** — French eval is a language-mismatch baseline (model card explicitly states it does not support fr/de/es transcription, only en). Useful as a starting point for measuring future fine-tuning gain.
- `voxtral_small4bit_fr`: Run in 4-bit (NF4) via bitsandbytes due to L4 23 GB VRAM cap. Quantization typically adds ~0.2-0.5 pt WER vs fp16.

### zh-CN Common Voice 21 (500 test utterances)

| Tag | Model | Family | Params | CER | Wall (s) |
|---|---|---|---:|---:|---:|
| `tiny_zh` | `openai/whisper-tiny` | whisper | 38 M | 59.63% | 19 |
| `base_zh` | `openai/whisper-base` | whisper | 73 M | 49.55% | 18 |
| `sensevoice_zh` | `FunAudioLLM/SenseVoiceSmall` | sensevoice | 234 M | 11.87% | 30 |
| `small_zh` | `openai/whisper-small` | whisper | 242 M | 33.52% | 28 |
| `medium_zh` | `openai/whisper-medium` | whisper | 764 M | 28.73% | 78 |
| `turbo_zh` | `openai/whisper-large-v3-turbo` | whisper | 809 M | 17.28% | 48 |
| `largev2_zh` | `openai/whisper-large-v2` | whisper | 1.54 B | 28.33% | 142 |
| `belle_zh` | `BELLE-2/Belle-whisper-large-v3-zh` | whisper | 1.54 B | 13.13% | 120 |
| `largev3_zh` | `openai/whisper-large-v3` | whisper | 1.54 B | 17.60% | 139 |

