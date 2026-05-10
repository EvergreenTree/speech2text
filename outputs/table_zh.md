| Run | Trainable | Wall eval (s) | CER | Δ CER abs | Δ CER rel |
|---|---:|---:|---:|---:|---:|
| Whisper-tiny **baseline** | — | 21.6 | **0.5938** | — | — |
| Whisper-tiny + LoRA (recette zh, LR 1e-4, 1 ep) | 1,5 M | 14.8 | **0.4168** | -0.1770 | -29.8 % |
| Whisper-tiny **full FT** | 39 M | 16.6 | **0.3551** | -0.2387 | -40.2 % |
| Whisper-small **baseline** | — | 20.9 | **0.3352** | — | — |
| Whisper-small + LoRA (recette zh, LR 1e-4, 1 ep) | 7,1 M | 25.8 | **0.2322** | -0.1030 | -30.7 % |
| Whisper-small **full FT** | 244 M | 27.0 | **0.2208** | -0.1144 | -34.1 % |
| Whisper-medium **baseline** | — | 62.2 | **0.2873** | — | — |
| Whisper-medium + LoRA (recette zh, LR 1e-4, 2 ep) | 18,9 M | 62.7 | **0.1322** | -0.1551 | -54.0 % |
