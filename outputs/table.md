| Run | Trainable | Wall eval (s) | WER | CER | Δ WER abs | Δ WER rel |
|---|---:|---:|---:|---:|---:|---:|
| Whisper-small **baseline** | — | 31.8 | **0.1386** | 0.0508 | — | — |
| Whisper-small + LoRA (LR 1e-4, 1 ep, recette zh) | 7,1 M | 34.1 | **0.1551** | 0.0600 | +0.0165 | +11.9 % |
| Whisper-small + LoRA (LR 3e-5, 2 ep, **recette fr**) | 7,1 M | 30.6 | **0.1547** | 0.0571 | +0.0161 | +11.6 % |
| Whisper-small **full FT** | 244 M | 32.8 | **0.1456** | 0.0549 | +0.0070 | +5.0 % |
| Whisper-small **from scratch** (random init, 10 ep) | 244 M | 17.1 | **0.9615** | 0.7593 | +0.8229 | +593.7 % |
| Whisper-medium **baseline** | — | 88.6 | **0.0820** | 0.0292 | — | — |
| Whisper-medium + LoRA (recette fr) | 18,9 M | 88.3 | **0.0859** | 0.0307 | +0.0039 | +4.8 % |
| Whisper-large-v3-turbo **baseline** | — | 63.6 | **0.0581** | 0.0199 | — | — |
| Whisper-large-v3-turbo + LoRA (recette fr) | ~12 M | 62.1 | **0.0617** | 0.0213 | +0.0036 | +6.2 % |
| _ref_ : wav2vec2-CTC-français (zero-shot, paradigme CTC) | — | 13.5 | **0.1037** | 0.0465 | — | — |
| _ref_ : Whisper-large-v3 distil-fr-dec4 (zero-shot) | — | 64.1 | **0.0697** | 0.0256 | — | — |
