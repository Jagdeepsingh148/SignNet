| Run | Input | Val acc (%) | Test acc (%) | Macro-F1 | asl_alphabet_test (%) | asl_alphabet_test + mirror TTA (%) | Params (M) | Latency (ms) | Train (min) |
|---|---|---|---|---|---|---|---|---|---|
| effb0_crop | MediaPipe hand crop | 99.17 | 98.90 | 0.9890 | 50.0 (13/26) | 61.5 (16/26) | 4.04 | 12.4 | 9.9 |
| effb0_crop_robust | MediaPipe hand crop + robust aug | 99.36 | 99.36 | 0.9935 | 65.4 (17/26) | 65.4 (17/26) | 4.04 | 13.9 | 7.4 |
| effb0_full | Full image | 99.63 | 99.45 | 0.9945 | 69.2 (18/26) | 65.4 (17/26) | 4.04 | 13.1 | 5.9 |

Dataset: Marxulia/asl_sign_languages_alphabets_v03 (split train/val/test = 8698/1087/1088, stratified, seed 42).
Model: google/efficientnet-b0 fine-tuned; latency = batch-1 forward pass on NVIDIA GeForce RTX 3050 Laptop GPU.
