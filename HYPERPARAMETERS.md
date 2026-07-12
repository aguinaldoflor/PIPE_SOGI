# PIPE-SOGI — Hyperparameter Search Spaces and Final Configurations

This file documents the hyperparameter search spaces used in the Optuna-based tuning procedure described in the paper *"PIPE-SOGI: A Multi-Model Framework for Steel Price Forecasting and Financial Risk Assessment in the Oil & Gas Industry."* All Deep Learning models (RNN, Transformer, N-HiTS, TCN, TSMixer, DLinear) were tuned using these ranges, applied consistently across all four steel-price datasets (China, USA, Germany, Turkey).

Format: `Parameter — Search Space (or fixed choices)`. Ranges given as `[min - max, step = s]` denote a stepped grid; ranges given as `[a, b]` with no step denote a continuous or categorical choice as indicated.

## RNN

| Hyperparameter | Tuning Range |
|---|---|
| Model Type | [LSTM, GRU] |
| RNN Layers | [1, 2] |
| Input Chunk | [189 – 315, step = 21] |
| Training Length | [Input Chunk + 1, Input Chunk + 63, step = 7] |

## Transformer

| Hyperparameter | Tuning Range |
|---|---|
| Input Chunk | [32 – 124, step = 32] |
| Dim Feedforward | [256, 512, 1024] |
| Dimension | [64, 96, 128] |
| Heads | [4, 8] |
| Encoder Layers | [2 – 6] |
| Decoder Layers | [2 – 6] |

## N-HiTS

| Hyperparameter | Tuning Range |
|---|---|
| Input Chunk | [32 – 128, step = 32] |
| Layers | [128, 256, 512] |
| Stacks | [2 – 4] |
| Blocks | [1 – 3] |

## TCN

| Hyperparameter | Tuning Range |
|---|---|
| Input Chunk | [32 – 128, step = 32] |
| Kernel Size | [3 – 7, step = 2] |
| Filters | [4 – 10, step = 2] |
| Layers | [2 – 5] |
| Weight Decay | [1e-7, 1e-4] |

## TSMixer

| Hyperparameter | Tuning Range |
|---|---|
| Input Chunk | [12 – 64, step = 32] |
| Hidden Dim | [128, 256, 512] |
| Feed-forward Layer | [128, 256, 512] |
| Blocks | [1, 4] |

## DLinear

| Hyperparameter | Tuning Range |
|---|---|
| Input Chunk | [60 – 252, step = 30] |
| Kernel | [5 – 25, step = 2] |
| Shared Weights | [True, False] |

## All Models (shared search space)

| Hyperparameter | Tuning Range | Applies to |
|---|---|---|
| Output Chunk | [8% – 25% of input chunk length] | All except RNN |
| Dropout | [0.05 – 0.30] | All except DLinear |
| Learning Rate | [1e-4 – 5e-3] | All models |
| Weight Decay | [1e-7, 1e-4] | All models except TCN (listed separately above) |
| Loss Function | [MAE, MSE, HuberLoss] | All models |
| Likelihood | [True, False] | All models |
| Activation | [relu, gelu] | All except RNN |

## Notes

- Optimization was performed independently per dataset/model pair using Optuna's Bayesian search (TPE sampler).
- The final selected configuration per model/dataset (a single point within the ranges above) is stored in the corresponding notebook (`PIPESOGI_001_CHI.ipynb`, `PIPESOGI_002_USA.ipynb`, `PIPESOGI_003_GER.ipynb`, `PIPESOGI_004_TUR.ipynb`) and in the `output/` and `result/` directories of this repository.
- **Two items need confirmation before this file is treated as final:**
  1. **N-HiTS "Blocks"**: the original manuscript table had this as "Blocis," which is not a standard term — corrected here to "Blocks" (stack depth parameter). Confirm this is what was intended.
  2. **"All Models" row for Learning Rate / Weight Decay**: the original table's row read `Learning Rate | & [1e-4 - 5e-3] | [1e-7, 1e-4]` with a blank second parameter name. Since the second range `[1e-7, 1e-4]` is identical to TCN's dedicated Weight Decay range, this file assumes the missing label is **Weight Decay** (applying to all non-TCN models). Please confirm this inference — if the second column meant something else, the table needs a correction, not just a relabeling.
