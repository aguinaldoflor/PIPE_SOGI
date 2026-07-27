# PIPE-SOGI — Hyperparameter Search Spaces and Training Protocol

This file documents the hyperparameter search spaces and the training protocol used
in the Optuna-based tuning procedure of *"PIPE-SOGI: Risk-Consistent Deep Learning
for Steel Price Value-at-Risk and Expected Shortfall in Oil & Gas Procurement."*

All Deep Learning models (RNN, Transformer, N-HiTS, TCN, TSMixer, DLinear) were
tuned with the ranges below, applied **identically across the four steel-price
datasets** (China, USA, Germany, Türkiye) so that cross-market differences reflect
market structure rather than search-space asymmetry.

Generated from `pipeline_v2p.py` (probabilistic arm). Format:
`Parameter — Search Space`; `[min – max, step = s]` = stepped integer grid;
`[a, b] (log)` = continuous log-uniform; `[x, y, z]` = categorical.

---

## 1. Tuning objective and protocol

| Item | Setting |
|---|---|
| **Objective minimized** | **Mean pinball (quantile) loss** over the trained quantiles, on the **validation set** |
| Sampler | Optuna TPE (`TPESampler`, seed = 42) |
| Pruner | `MedianPruner` (`n_warmup_steps = 5`) |
| Trials | **50 per model per market** |
| Epochs per trial | 150 (final refit: up to 500, early-stopped) |
| Data split | Temporal **65 % / 15 % / 20 %** (train / validation / test) |
| Forecast protocol | One-shot `historical_forecasts`, direct **h-step** (primary cell **h = 5**) |
| Seeds (final refit) | 42, 43, 44 — median seed reported by pinball, dispersion exported |
| Hardware | NVIDIA RTX 4070 Ti Super (GPU accelerator) |

**Note on the objective.** Tuning, training and epoch selection all minimize the
same quantile-consistent loss: the torch models are trained with a
`QuantileRegression` likelihood, `EarlyStopping` monitors `val_loss` (which *is*
the pinball loss under that likelihood), and Optuna scores each trial by the
validation pinball. Point metrics (SMAPE, RMSE, MASE) are reported as secondary
diagnostics and never drive selection. Final model ranking uses the FZ0
(Fissler–Ziegel) score of the risk layer, not the tuning objective.

---

## 2. Fixed (non-tuned) probabilistic settings

Unified across every torch model, so the quantile heads are comparable:

| Item | Value |
|---|---|
| Likelihood | `QuantileRegression` (fixed — **not** searched) |
| Quantiles | **[0.05, 0.50, 0.95]** (95 % upper-tail focus; 90 % central interval) |
| Loss function | `None` (the likelihood defines the training loss) |
| Output chunk length | = forecast horizon `h` (direct multi-horizon) |
| LR scheduler | `ReduceLROnPlateau` |

---

## 3. Per-model search spaces

### RNN (BlockRNNModel)

| Hyperparameter | Search space |
|---|---|
| Input Chunk Length | [42 – 126, step = 21] |
| Model Type | [LSTM, GRU] |
| Hidden Dimension | [16 – 128, step = 16] |
| RNN Layers | [1 – 4] |
| Dropout | [0.0 – 0.4] |
| Learning Rate | [1e-4 – 5e-3] (log) |
| Weight Decay | [1e-7 – 1e-4] (log) |

### Transformer

| Hyperparameter | Search space |
|---|---|
| Input Chunk Length | [42 – 126, step = 21] |
| d_model / n_heads (joint) | [64/4, 96/4, 128/4, 128/8] |
| Encoder Layers | [2 – 6] |
| Decoder Layers | [2 – 6] |
| Dim Feedforward | [256, 512, 1024] |
| Dropout | [0.1 – 0.5] |
| Activation | [relu, gelu] |
| Learning Rate | [1e-4 – 5e-3] (log) |
| Weight Decay | [1e-7 – 1e-4] (log) |

> `d_model` and `n_heads` are searched as a single categorical pair to avoid
> invalid combinations (`d_model` must be divisible by `n_heads`).

### N-HiTS

| Hyperparameter | Search space |
|---|---|
| Input Chunk Length | [32 – 128, step = 32] |
| Stacks | [2 – 4] |
| Blocks per stack | [1 – 3] |
| Layers per block | [1 – 3] |
| Layer Width | [128, 256, 512] |
| Freq. downsampling (per stack) | [1 – 4] for all but the last stack (last fixed at 1) |
| Dropout | [0.0 – 0.3] |
| Activation | [ReLU, GELU] |
| Learning Rate | [1e-4 – 5e-3] (log) |
| Weight Decay | [1e-7 – 1e-4] (log) |

### TCN

| Hyperparameter | Search space |
|---|---|
| Input Chunk Length | [32 – 128, step = 32] |
| Kernel Size | [3 – 7, step = 2] |
| Filters | [4 – 10, step = 2] |
| Layers | [2 – 5] |
| Dropout | [0.0 – 0.4] |
| Learning Rate | [1e-4 – 5e-3] (log) |
| Weight Decay | [1e-7 – 1e-4] (log) |

> Fixed: `dilation_base = 2`, `weight_norm = True`. If the sampled input chunk is
> not greater than `h`, it is raised to `h + input_chunk_length` (TCN requires
> input chunk > output chunk).

### TSMixer

| Hyperparameter | Search space |
|---|---|
| Input Chunk Length | [12 – 64, step = 32] |
| Hidden Size | [128, 256, 512] |
| Blocks | [1 – 4] |
| Feed-forward Size | [128, 256, 512] |
| Dropout | [0.1 – 0.5] |
| Activation | [ReLU, GELU] |
| Normalization | [LayerNorm, LayerNormNoBias] |
| Learning Rate | [1e-4 – 5e-3] (log) |
| Weight Decay | [1e-7 – 1e-4] (log) |

### DLinear

| Hyperparameter | Search space |
|---|---|
| Input Chunk Length | [60 – 252, step = 30] |
| Kernel Size | [5 – 25, step = 2] |
| Shared Weights | [True, False] |
| Constant Initialization | [True, False] |
| Learning Rate | [1e-5 – 1e-2] (log) |
| Weight Decay | [1e-6 – 1e-3] (log) |

> DLinear has no dropout or activation parameters (linear decomposition model);
> its LR/WD ranges are wider because the model is far smaller than the others.

---

## 4. Fixed training settings per model

| Model | Batch size | EarlyStopping patience | min_delta | LR-scheduler (patience, factor) |
|---|---|---|---|---|
| RNN | 64 | 15 | 0.001 | (10, 0.2) |
| Transformer | 64 | 25 | 0.001 | (25, 0.2) |
| N-HiTS | 64 | 25 | 0.001 | (25, 0.2) |
| TCN | 64 | 25 | 0.001 | (25, 0.2) |
| TSMixer | 32 | 25 | 0.001 | (25, 0.2) |
| DLinear | 32 | 15 | 0.0005 | (15, 0.5) |

Patience is scaled to model size: small, fast-converging models (RNN, DLinear)
stop earlier; heavier architectures are given more epochs to escape loss plateaus.
These settings are **identical in both arms** of the study, so the point-vs-
probabilistic comparison isolates the effect of the objective, not of the
training schedule.

---

## 5. Covariates and statistical baselines

Past covariates (identical for every model that supports them — DLinear is
univariate by design): lags 1, 5, 22, 66 of the scaled target, plus rolling
standard deviations over 5 and 22 days (`vol5`, `vol22`). Target and covariates
are log1p-transformed and scaled with a `RobustScaler` **fitted on the training
split only**.

Statistical baselines are not tuned: ARIMA(5,1,5), Exponential Smoothing
(Holt–Winters), and `LinearRegressionModel(lags = 66)`, all evaluated under the
same one-shot protocol. Benchmarks: Naive-h and Drift-h.

---

## Reproducibility

- Caches: `result/v2p_hp_<market>.xlsx` (one sheet per model/horizon) store the
  selected configuration and its validation pinball; re-runs read the cache
  instead of re-tuning (`force_tune = True` overrides).
- Per-horizon checkpoints: `result/v2p_ckpt_<market>.xlsx` (crash-safety).
- Results: `result/v2p_result_steel_<market>.xlsx` — sheets `summary_*`,
  `rank_prob` (pinball/Winkler ranking), `rank` (point ranking), `error_*_h5`
  (per-model residuals and native quantiles feeding the risk layer).
- Point-arm caches (`v2_*`) are **not** reused by the probabilistic arm: the
  search space and objective changed, so previously tuned configurations are not
  transferable.
