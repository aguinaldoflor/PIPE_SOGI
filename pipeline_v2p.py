"""
PIPE-SOGI — Pipeline V2p (PROBABILISTIC arm: native quantile forecasting)
=========================================================================
Derived from pipeline_v2.py (point arm, kept intact as the baseline chapter).
Requires darts >= 0.25 (predict_likelihood_parameters in historical_forecasts).

Changes vs. V2 (each tied to the thesis pivot: model the conditional
distribution, not the point):
1. `use_likelihood` and `loss_fn` REMOVED from the search space; every torch
   model trains with a fixed, unified QuantileRegression on QUANTILES =
   [0.05, 0.5, 0.95] (95% CI focus; unifies the old TCN/other inconsistency).
2. Optuna objective = mean pinball loss over the three quantiles on the
   validation set (same one-shot protocol). EarlyStopping monitors val_loss,
   which IS the pinball now — training and selection are aligned end to end.
3. `_hf` uses predict_likelihood_parameters=True: deterministic quantile
   heads, no sampling, no collapse to the median. Monotonic rearrangement
   guards against quantile crossing. `inverse_chain` is applied per quantile
   (valid: quantiles are equivariant under monotone transforms).
4. Exports: `{pfx}_pred` (median, backward compatible) + `{pfx}_q05` and
   `{pfx}_q95` for every model with native quantiles (DL + Vincent ensembles).
   The VaR layer consumes q95 directly (buyer risk = upper price tail).
5. Ensembles: EnsMean / EnsDynSMAPE / EnsPinballDyn combine quantiles by
   Vincentization (their time-varying weights applied to q05/q95 as well).
   EnsRegTop3 / EnsResidChamp / EnsShrink remain point-only (baseline arm).
   EnsPinballDyn weighting quantile moved 0.01 -> 0.05 (95% CI coherence).
6. Ranking: primary = PB05, PB95, Winkler90 (+ Cover90 reported); secondary =
   SMAPE/RMSE/MASE of the median (kept for FRI compatibility). Two rank
   sheets are exported: `rank_prob` (quantile-capable models) and `rank`
   (point ranking over all models, as before).

All artifacts use the `v2p_` prefix (hp cache, checkpoints, results) so the
baseline V2 artifacts are never touched or reused: search space and objective
changed, therefore old hp caches are methodologically non-transferable.
"""
import os
import ast as _ast
import time
import random
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from darts import TimeSeries
from darts.models import (BlockRNNModel, TransformerModel, NHiTSModel, DLinearModel,
                          TCNModel, TSMixerModel, ARIMA, ExponentialSmoothing,
                          LinearRegressionModel)
from darts.utils.likelihood_models.torch import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import norm, chi2

EPS = 1e-12
SPLITS = (0.65, 0.15, 0.20)
H_LIST = [1, 5, 21, 66]
SEEDS = [42, 43, 44]
N_TRIALS = 50
QUANTILES = [0.05, 0.5, 0.95]          # fixed, unified — 95% CI focus
ALPHA = 0.10                           # 1 - (0.95 - 0.05) interval level
TORCH_KWARGS = {"pl_trainer_kwargs": {"accelerator": "gpu", "devices": [0]}}
DL_MODELS = ["RNN", "Transformer", "N-HiTS", "DLinear", "TCN", "TSMixer"]
STAT_MODELS = ["ARIMA", "ExpSmoothing", "LinearRegression"]

optuna.logging.set_verbosity(optuna.logging.WARNING)


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. PREPROCESSING + THREE-WAY SPLIT + DIAGNOSTICS  (unchanged vs V2)
# ============================================================
def preprocess(target_col, dataset_path="dataset/ds_steel.xlsx", splits=SPLITS,
               oil_path=None):
    """Optional oil_path: appends brent/wti/opec (log-levels + 22d returns) to the
    past-covariates stack — used ONLY in the ablation study (M6); the main run is
    univariate. When provided, the sample is trimmed at the last real oil quote."""
    df = pd.read_excel(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    df_b = df.asfreq("B")
    interp_share = float(df_b[target_col].isna().mean())
    stale_raw = float(df_b[target_col].dropna().diff().eq(0).mean())

    df = df_b.reset_index().interpolate()
    q01, q99 = df[target_col].quantile([0.01, 0.99])
    df[target_col] = np.clip(df[target_col], q01, q99)

    oil_log = None
    if oil_path is not None:
        oil = pd.read_excel(oil_path)
        oil["date"] = pd.to_datetime(oil["date"])
        oil = oil.set_index("date")[["brent", "wti", "opec"]]
        oil_end = oil.dropna(how="all").index.max()
        df = df[df["date"] <= oil_end].copy()
        oil = oil.reindex(pd.DatetimeIndex(df["date"])).interpolate(limit_direction="both")
        oil = oil.clip(lower=0.01)   # WTI closed negative on 2020-04-20: log1p(x<-1)=NaN
        oil_log = np.log1p(oil)
        for c in ["brent", "wti", "opec"]:
            oil_log[f"{c}_ret22"] = oil_log[c].diff(22).fillna(0.0)

    series_raw = TimeSeries.from_dataframe(df, time_col="date",
                                           value_cols=target_col, freq="B")
    series_trans = series_raw.map(np.log1p)

    n = len(series_trans)
    t1 = int(n * splits[0])
    t2 = int(n * (splits[0] + splits[1]))

    scaler = Scaler(scaler=RobustScaler())
    scaler.fit(series_trans[:t1])
    series_scaled = scaler.transform(series_trans)

    base = series_scaled.components[0]
    covs = []
    for l in [1, 5, 22, 66]:
        covs.append(series_scaled.shift(l).with_columns_renamed([base], [f"lag_{l}"]))
    for w, nm in [(5, "vol5"), (22, "vol22")]:
        v = pd.Series(series_scaled.univariate_values()).rolling(w, min_periods=1).std().values
        covs.append(TimeSeries.from_times_and_values(series_scaled.time_index, v, columns=[nm]))
    if oil_log is not None:
        covs.append(TimeSeries.from_times_and_values(
            series_scaled.time_index, oil_log.values, columns=list(oil_log.columns)))
    cs = max(s.start_time() for s in covs); ce = min(s.end_time() for s in covs)
    covs = [s.slice(cs, ce) for s in covs]
    covariates = covs[0]
    for s in covs[1:]:
        covariates = covariates.stack(s)

    ssa = series_scaled.slice(covariates.start_time(), covariates.end_time())
    sra = series_raw.slice(covariates.start_time(), covariates.end_time())
    assert not np.isnan(covariates.values()).any(), "NaN in covariates after alignment cut!"

    n = len(ssa)
    t1 = int(n * splits[0]); t2 = int(n * (splits[0] + splits[1]))

    scaler_covs = Scaler(scaler=RobustScaler())
    scaler_covs.fit(covariates[:t1])
    covs_scaled = scaler_covs.transform(covariates)

    return dict(scaler=scaler, ssa=ssa, sra=sra, covs=covs_scaled,
                t1=t1, t2=t2, n=n,
                train_s=ssa[:t1], val_s=ssa[t1:t2], test_s=ssa[t2:],
                train_o=sra[:t1], val_o=sra[t1:t2], test_o=sra[t2:],
                interp_share=interp_share, stale_raw=stale_raw,
                target_col=target_col)


def inverse_chain(ts_scaled, scaler):
    return scaler.inverse_transform(ts_scaled).map(np.expm1).map(lambda v: np.maximum(v, EPS))


# ============================================================
# 2. MODELS: SEARCH SPACES (ocl = h FIXED) + BUILDERS
#    v2p: use_likelihood / loss_fn removed from every space
# ============================================================
def _suggest(kind, trial):
    hp = {}
    if kind == "RNN":
        hp["input_chunk_length"] = trial.suggest_int("input_chunk_length", 42, 126, step=21)
        hp["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        hp["rnn_model"] = trial.suggest_categorical("rnn_model", ["LSTM", "GRU"])
        hp["hidden_dim"] = trial.suggest_int("hidden_dim", 16, 128, step=16)
        hp["n_rnn_layers"] = trial.suggest_int("n_rnn_layers", 1, 4)
        hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
    elif kind == "Transformer":
        hp["input_chunk_length"] = trial.suggest_int("input_chunk_length", 42, 126, step=21)
        hp["dm_nh"] = trial.suggest_categorical("dm_nh", ["64_4", "96_4", "128_4", "128_8"])
        hp["encoder_layers"] = trial.suggest_int("encoder_layers", 2, 6)
        hp["decoder_layers"] = trial.suggest_int("decoder_layers", 2, 6)
        hp["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [256, 512, 1024])
        hp["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        hp["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        hp["activation"] = trial.suggest_categorical("activation", ["relu", "gelu"])
    elif kind == "N-HiTS":
        hp["input_chunk_length"] = trial.suggest_int("input_chunk_length", 32, 128, step=32)
        hp["num_stacks"] = trial.suggest_int("num_stacks", 2, 4)
        hp["num_blocks"] = trial.suggest_int("num_blocks", 1, 3)
        hp["num_layers_in_block"] = trial.suggest_int("num_layers_in_block", 1, 3)
        hp["layer_width"] = trial.suggest_categorical("layer_width", [128, 256, 512])
        nfd = []
        for i in range(hp["num_stacks"]):
            f = (trial.suggest_int(f"n_freq_downsample_stack_{i}", 1, 4)
                 if i < hp["num_stacks"] - 1 else 1)
            nfd.append([f] * hp["num_blocks"])
        hp["n_freq_downsample"] = str(nfd)
        hp["activation"] = trial.suggest_categorical("activation", ["ReLU", "GELU"])
        hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        hp["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    elif kind == "DLinear":
        hp["input_chunk_length"] = trial.suggest_int("input_chunk_length", 60, 252, step=30)
        hp["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        hp["kernel_size"] = trial.suggest_int("kernel_size", 5, 25, step=2)
        hp["shared_weights"] = trial.suggest_categorical("shared_weights", [False, True])
        hp["const_init"] = trial.suggest_categorical("const_init", [True, False])
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    elif kind == "TCN":
        hp["input_chunk_length"] = trial.suggest_int("input_chunk_length", 32, 128, step=32)
        hp["kernel_size"] = trial.suggest_int("kernel_size", 3, 7, step=2)
        hp["num_filters"] = trial.suggest_int("num_filters", 4, 10, step=2)
        hp["num_layers"] = trial.suggest_int("num_layers", 2, 5)
        hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.4)
        hp["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    elif kind == "TSMixer":
        hp["input_chunk_length"] = trial.suggest_int("input_chunk_length", 12, 64, step=32)
        hp["hidden_size"] = trial.suggest_categorical("hidden_size", [128, 256, 512])
        hp["num_blocks"] = trial.suggest_int("num_blocks", 1, 4)
        hp["ff_size"] = trial.suggest_categorical("ff_size", [128, 256, 512])
        hp["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        hp["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        hp["activation"] = trial.suggest_categorical("activation", ["ReLU", "GELU"])
        hp["norm_type"] = trial.suggest_categorical("norm_type", ["LayerNorm", "LayerNormNoBias"])
    return hp


def build_model(kind, hp, h, seed, n_epochs=500):
    """v2p: fixed QuantileRegression on QUANTILES for every model; loss_fn=None
    (training loss = pinball via the likelihood)."""
    lik = QuantileRegression(quantiles=QUANTILES)
    icl = int(hp["input_chunk_length"])
    if kind == "TCN" and icl <= h:
        icl = h + icl  # TCN requires icl > ocl
    pat = 15 if kind in ("RNN", "DLinear") else 25
    md = 0.0005 if kind == "DLinear" else 0.001
    stopper = EarlyStopping(monitor="val_loss", patience=pat, min_delta=md,
                            mode="min", verbose=False)
    pl = TORCH_KWARGS["pl_trainer_kwargs"].copy(); pl["callbacks"] = [stopper]
    common = dict(input_chunk_length=icl, output_chunk_length=h, n_epochs=n_epochs,
                  random_state=seed, force_reset=True, save_checkpoints=False,
                  likelihood=lik, loss_fn=None,
                  optimizer_kwargs={"lr": float(hp["lr"]),
                                    "weight_decay": float(hp.get("weight_decay", 0.0))},
                  lr_scheduler_cls=ReduceLROnPlateau,
                  pl_trainer_kwargs=pl)
    if kind == "RNN":
        return BlockRNNModel(model=str(hp["rnn_model"]), hidden_dim=int(hp["hidden_dim"]),
            n_rnn_layers=int(hp["n_rnn_layers"]), dropout=float(hp["dropout"]),
            batch_size=64, lr_scheduler_kwargs={"patience": 10, "factor": 0.2}, **common)
    if kind == "Transformer":
        dm, nh = str(hp["dm_nh"]).split("_")
        return TransformerModel(d_model=int(dm), nhead=int(nh),
            num_encoder_layers=int(hp["encoder_layers"]),
            num_decoder_layers=int(hp["decoder_layers"]),
            dim_feedforward=int(hp["dim_feedforward"]), dropout=float(hp["dropout"]),
            activation=str(hp["activation"]), batch_size=64,
            lr_scheduler_kwargs={"patience": 25, "factor": 0.2}, **common)
    if kind == "N-HiTS":
        return NHiTSModel(num_stacks=int(hp["num_stacks"]), num_blocks=int(hp["num_blocks"]),
            num_layers=int(hp["num_layers_in_block"]), layer_widths=int(hp["layer_width"]),
            n_freq_downsample=_ast.literal_eval(str(hp["n_freq_downsample"])),
            dropout=float(hp["dropout"]), activation=str(hp["activation"]), batch_size=64,
            lr_scheduler_kwargs={"patience": 25, "factor": 0.2}, **common)
    if kind == "DLinear":
        return DLinearModel(kernel_size=int(hp["kernel_size"]),
            shared_weights=bool(hp["shared_weights"]), const_init=bool(hp["const_init"]),
            batch_size=32, lr_scheduler_kwargs={"patience": 15, "factor": 0.5}, **common)
    if kind == "TCN":
        return TCNModel(kernel_size=int(hp["kernel_size"]), num_filters=int(hp["num_filters"]),
            num_layers=int(hp["num_layers"]), dilation_base=2, weight_norm=True,
            dropout=float(hp["dropout"]), batch_size=64,
            lr_scheduler_kwargs={"patience": 25, "factor": 0.2}, **common)
    if kind == "TSMixer":
        return TSMixerModel(hidden_size=int(hp["hidden_size"]), ff_size=int(hp["ff_size"]),
            num_blocks=int(hp["num_blocks"]), dropout=float(hp["dropout"]),
            activation=str(hp["activation"]), norm_type=str(hp["norm_type"]), batch_size=32,
            lr_scheduler_kwargs={"patience": 25, "factor": 0.2}, **common)
    raise ValueError(kind)


USES_COVS = {"RNN": True, "Transformer": True, "N-HiTS": True,
             "DLinear": False, "TCN": True, "TSMixer": True}


# ============================================================
# 3. FORECASTING (native quantile heads) AND METRICS
# ============================================================
def _hf(model, d, h, start_ts, end_slice, uses_covs):
    """One-shot historical_forecasts with ocl=h. v2p: deterministic quantile
    heads via predict_likelihood_parameters (no sampling, no median collapse).
    Returns {q: TimeSeries in price space} for q in QUANTILES.
    Monotonic rearrangement (row-wise sort) prevents quantile crossing; the
    inverse chain preserves quantiles (monotone transform equivariance)."""
    kw = dict(series=d["ssa"][:end_slice] if end_slice else d["ssa"],
              start=start_ts, forecast_horizon=h, stride=1,
              retrain=False, last_points_only=True, verbose=False,
              predict_likelihood_parameters=True)
    if uses_covs:
        kw["past_covariates"] = d["covs"]
    ps = model.historical_forecasts(**kw)
    vals = ps.values()                      # (n, len(QUANTILES)), asc. quantile order
    vals = np.sort(vals, axis=1)            # crossing fix
    out = {}
    for i, q in enumerate(QUANTILES):
        ts_q = TimeSeries.from_times_and_values(ps.time_index, vals[:, [i]],
                                                columns=[d["target_col"]])
        out[q] = inverse_chain(ts_q, d["scaler"])
    return out


def pinball(y, qh, q):
    e = y - qh
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def q_metrics(y, q05, q95, alpha=ALPHA):
    """Interval metrics for the 90% central interval [q05, q95]."""
    if len(y) == 0:
        return dict(PB05=np.nan, PB95=np.nan, Cover90=np.nan, Winkler90=np.nan)
    cover = float(np.mean((y >= q05) & (y <= q95)))
    wink = float(np.mean((q95 - q05)
                         + (2 / alpha) * np.where(y < q05, q05 - y, 0.0)
                         + (2 / alpha) * np.where(y > q95, y - q95, 0.0)))
    return dict(PB05=pinball(y, q05, 0.05), PB95=pinball(y, q95, 0.95),
                Cover90=cover, Winkler90=wink)


def np_metrics(y, yh, scale_h):
    e = y - yh; ae = np.abs(e)
    return dict(MAE=float(ae.mean()), RMSE=float(np.sqrt((e**2).mean())),
                SMAPE=float(np.mean(100*(2*ae/(np.abs(y)+np.abs(yh)+EPS)))),
                MASE=float(ae.mean()/scale_h))


def align_q(qs, ref_ts):
    """Aligns the quantile dict (shared time index) to the reference series.
    Returns y, {q: np.array}, time_index."""
    p = qs[0.5].slice_intersect(ref_ts)
    r = ref_ts.slice_intersect(p)
    Q = {q: ts.slice_intersect(r).values().flatten() for q, ts in qs.items()}
    return r.values().flatten(), Q, p.slice_intersect(r).time_index


def align_to(pred_ts, ref_ts):
    p = pred_ts.slice_intersect(ref_ts)
    r = ref_ts.slice_intersect(p)
    return r.values().flatten(), p.values().flatten(), p.time_index


# ============================================================
# 4. OPTUNA ON THE VALIDATION SET (objective = mean pinball)
# ============================================================
def tune(kind, d, h, n_trials=N_TRIALS, seed=42, n_epochs_trial=150):
    def objective(trial):
        hp = _suggest(kind, trial)
        set_seed(seed)
        model = build_model(kind, hp, h, seed, n_epochs=n_epochs_trial)
        fit_kw = dict(series=d["train_s"], val_series=d["val_s"], verbose=False)
        if USES_COVS[kind]:
            fit_kw.update(past_covariates=d["covs"][:d["t1"]],
                          val_past_covariates=d["covs"][:d["t2"]])
        model.fit(**fit_kw)
        # evaluation ONLY on validation; series truncated at t2 => test invisible
        qs = _hf(model, d, h, d["val_s"].start_time(), d["t2"], USES_COVS[kind])
        y, Q, _ = align_q(qs, d["val_o"])
        if len(y) == 0 or any(np.isnan(Q[q]).all() for q in QUANTILES):
            return float("inf")            # divergent trial -> pruned away
        return float(np.mean([pinball(y, Q[q], q) for q in QUANTILES]))

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=seed),
                                pruner=MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    hp = dict(study.best_trial.params)
    if kind == "N-HiTS":  # rebuild list of lists (per-stack x per-block)
        facts = [hp.pop(f"n_freq_downsample_stack_{i}")
                 for i in range(hp["num_stacks"] - 1)] + [1]
        hp["n_freq_downsample"] = str([[f] * hp["num_blocks"] for f in facts])
    return hp, float(study.best_value)


# ============================================================
# 5. FINAL RUN: refit on train+val -> test (multi-seed, quantile output)
# ============================================================
def final_runs(kind, hp, d, h, seeds=SEEDS):
    out = []
    for seed in seeds:
        set_seed(seed)
        model = build_model(kind, hp, h, seed)
        fit_kw = dict(series=d["ssa"][:d["t2"]], val_series=d["val_s"], verbose=False)
        if USES_COVS[kind]:
            fit_kw.update(past_covariates=d["covs"][:d["t2"]],
                          val_past_covariates=d["covs"][:d["t2"]])
        model.fit(**fit_kw)
        qs = _hf(model, d, h, d["test_s"].start_time(), None, USES_COVS[kind])
        y, Q, idx = align_q(qs, d["test_o"])
        out.append((seed, idx, y, Q))
    return out


def stat_baseline_runs(d, h):
    """ARIMA/ETS/LR under the SAME protocol (fit on train+val, hf on test).
    Point-only: they belong to the baseline arm; their tail risk continues to
    be modeled by the two-stage residual methods in the VaR layer."""
    res = {}
    fitsl = d["ssa"][:d["t2"]]
    models = {
        "ARIMA": (ARIMA(p=5, d=1, q=5), False),
        "ExpSmoothing": (ExponentialSmoothing(), True),   # non-transferable: needs retrain
        "LinearRegression": (LinearRegressionModel(lags=66, output_chunk_length=h), False),
    }
    for name, (m, needs_retrain) in models.items():
        try:
            t0 = time.time()
            m.fit(fitsl)
            ps = m.historical_forecasts(series=d["ssa"], start=d["test_s"].start_time(),
                                        forecast_horizon=h, stride=1,
                                        retrain=(21 if needs_retrain else False),
                                        last_points_only=True, verbose=False)
            pred = inverse_chain(ps, d["scaler"])
            y, yh, idx = align_to(pred, d["test_o"])
            res[name] = (idx, y, yh)
            print(f"  [{name}] ok ({round(time.time()-t0,1)}s)")
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
    return res


def naive_runs(d, h):
    full = d["sra"].values().flatten()
    idx_full = d["sra"].time_index
    pos = {dt: i for i, dt in enumerate(idx_full)}
    test_idx = d["test_o"].time_index[h-1:] if h > 1 else d["test_o"].time_index
    tgt = np.array([pos[dt] for dt in test_idx])
    y = full[tgt]
    naive = full[tgt - h]
    drift = np.array([full[o] + h*np.mean(np.diff(full[:o+1])) for o in (tgt - h)])
    scale_h = float(np.mean(np.abs(full[h:d["t1"]] - full[:d["t1"]-h])))
    return test_idx, y, {"NaiveH": naive, "DriftH": drift}, scale_h


# ============================================================
# 6. ENSEMBLES V2p (validation Top-3, temporal gap, Vincentization)
# ============================================================
def ensembles_v2(y, P_dict, Q05_dict, Q95_dict, top3, naive, h, window=66):
    """Point ensembles as in V2. EnsMean / EnsDynSMAPE / EnsPinballDyn also
    combine q05/q95 by applying the SAME time-varying weights to the member
    quantiles (Vincentization). EnsRegTop3 / EnsResidChamp / EnsShrink stay
    point-only (their mechanics are intrinsically point-based)."""
    P = np.column_stack([P_dict[m] for m in top3])
    A05 = np.column_stack([Q05_dict[m] for m in top3])
    A95 = np.column_stack([Q95_dict[m] for m in top3])
    n = len(y)
    gap = h

    def win(t):
        s1 = t - gap + 1; s0 = s1 - window
        return (s0, s1) if s0 >= 0 else None

    ens = {"EnsMean": P.mean(axis=1)}
    ens_q = {"EnsMean": (A05.mean(axis=1), A95.mean(axis=1))}

    dyn = np.zeros(n); pin = np.zeros(n)
    d05 = np.zeros(n); d95 = np.zeros(n)
    p05 = np.zeros(n); p95 = np.zeros(n)
    errors = y[:, None] - P
    def pb_arr(e, q): return (q - (e < 0).astype(float))*e
    for t in range(n):
        w_ = win(t)
        if w_ is None:
            dyn[t] = P[t].mean(); pin[t] = P[t].mean()
            d05[t] = A05[t].mean(); d95[t] = A95[t].mean()
            p05[t] = A05[t].mean(); p95[t] = A95[t].mean()
            continue
        s0, s1 = w_
        ry, rp = y[s0:s1], P[s0:s1]
        sm = np.mean(100*(2*np.abs(ry[:, None]-rp)/(np.abs(ry[:, None])+np.abs(rp)+EPS)), axis=0)+EPS
        wgt = (1/sm)/np.sum(1/sm)
        dyn[t] = np.sum(wgt*P[t]); d05[t] = np.sum(wgt*A05[t]); d95[t] = np.sum(wgt*A95[t])
        # v2p: weighting quantile 0.01 -> 0.05 (95% CI coherence)
        L = np.mean(pb_arr(errors[s0:s1], 0.05), axis=0)
        Ls = L/(np.median(np.abs(L))+EPS)
        z = -Ls/2.0; z -= z.max(); wq = np.exp(z); wq /= wq.sum()+EPS
        pin[t] = np.sum(wq*P[t]); p05[t] = np.sum(wq*A05[t]); p95[t] = np.sum(wq*A95[t])
    ens["EnsDynSMAPE"] = dyn; ens["EnsPinballDyn"] = pin
    ens_q["EnsDynSMAPE"] = (d05, d95); ens_q["EnsPinballDyn"] = (p05, p95)

    champ = P[:, 0]; champ_res = y - champ
    def _lag(res, k, nn):
        out = np.zeros(nn)
        if k < nn:
            out[k:] = res[:nn - k]
        return out

    n = len(y)
    lags = np.column_stack([_lag(champ_res, k, n) for k in range(h, h + 3)])

    X = np.column_stack([P, lags])
    reg = np.zeros(n); rc = np.zeros(n); shr = np.zeros(n)
    alphas = np.linspace(0, 1, 21)
    for t in range(n):
        w_ = win(t)
        if w_ is None:
            reg[t] = P[t].mean(); rc[t] = champ[t]; shr[t] = naive[t]; continue
        s0, s1 = w_
        r = Ridge(alpha=1.0); r.fit(X[s0:s1], y[s0:s1]); reg[t] = r.predict(X[t:t+1])[0]
        g = GradientBoostingRegressor(n_estimators=22, max_depth=3,
                                      learning_rate=0.05, random_state=42)
        g.fit(X[s0:s1], champ_res[s0:s1]); rc[t] = champ[t] + g.predict(X[t:t+1])[0]
        errs = [np.mean(np.abs(y[s0:s1] - (a*champ[s0:s1] + (1-a)*naive[s0:s1])))
                for a in alphas]
        a_star = alphas[int(np.argmin(errs))]
        shr[t] = a_star*champ[t] + (1-a_star)*naive[t]
    ens["EnsRegTop3"] = reg; ens["EnsResidChamp"] = rc; ens["EnsShrink"] = shr
    return ens, ens_q


# ============================================================
# 7. TESTS: DM (NW + stride) and Fisher  (on median errors)
# ============================================================
def dm_overlapping(e_m, e_n, L):
    dd = e_m**2 - e_n**2
    dbar, T = dd.mean(), len(dd)
    dc = dd - dbar
    s = np.mean(dc*dc)
    for l in range(1, L+1):
        s += 2*(1 - l/(L+1))*np.mean(dc[l:]*dc[:-l])
    stat = dbar/np.sqrt(s/T)
    return float(stat), float(norm.cdf(stat))

def dm_stride(e_m, e_n, h):
    dd = (e_m**2 - e_n**2)[::h]
    stat = dd.mean()/np.sqrt(dd.var(ddof=1)/len(dd))
    return float(stat), float(norm.cdf(stat))

def fisher(pvals):
    ps = np.clip(np.asarray(pvals, dtype=float), 1e-12, 1)
    X = -2*np.sum(np.log(ps))
    return float(X), float(1 - chi2.cdf(X, df=2*len(ps)))


# ============================================================
# 8. PER-COUNTRY ORCHESTRATOR (v2p artifacts)
# ============================================================
MODEL_PREFIX = {"RNN": "rnn", "Transformer": "trans", "N-HiTS": "nhits",
                "DLinear": "dlinear", "TCN": "tcn", "TSMixer": "tsmixer",
                "ARIMA": "arima", "ExpSmoothing": "es", "LinearRegression": "lr",
                "NaiveH": "naive", "DriftH": "drift",
                "EnsMean": "EnsMean", "EnsDynSMAPE": "EnsDynSMAPE",
                "EnsPinballDyn": "EnsPinballDyn", "EnsRegTop3": "EnsRegTop3",
                "EnsResidChamp": "EnsResidChamp", "EnsShrink": "EnsShrink"}


def _hp_cache_path(cc):
    return f"result/v2p_hp_{cc.lower()}.xlsx"


def load_or_tune(cc, kind, d, h, n_trials=N_TRIALS, force=False):
    """Hyperparameter cache in result/v2p_hp_<cc>.xlsx (sheet <kind>_h<h>).
    NOTE: v2 (point-arm) caches are intentionally NOT read — the search space
    and the objective changed, so they are not transferable."""
    path = _hp_cache_path(cc)
    sheet = f"{MODEL_PREFIX[kind]}_h{h}"
    if not force and os.path.exists(path):
        try:
            hp = pd.read_excel(path, sheet_name=sheet).iloc[0].to_dict()
            print(f"  [{kind}|h={h}] hp from cache")
            return hp
        except ValueError:
            pass
    print(f"  [{kind}|h={h}] Optuna ({n_trials} trials on validation, pinball)...")
    t0 = time.time()
    hp, best_val = tune(kind, d, h, n_trials=n_trials)
    hp["_val_pinball"] = best_val
    os.makedirs("result", exist_ok=True)
    mode = "a" if os.path.exists(path) else "w"
    kw = dict(engine="openpyxl", mode=mode)
    if mode == "a":
        kw["if_sheet_exists"] = "replace"
    with pd.ExcelWriter(path, **kw) as w:
        pd.DataFrame([hp]).to_excel(w, sheet_name=sheet, index=False)
    print(f"  [{kind}|h={h}] val_pinball={best_val:.5f} ({round(time.time()-t0,1)}s)")
    return hp


def _ckpt_path(cc, tag):
    sfx = f"_{tag}" if tag else ""
    return f"result/v2p_ckpt_{cc.lower()}{sfx}.xlsx"


def _load_checkpoint(cc, tag, h_list):
    path = _ckpt_path(cc, tag)
    rows_by_h, err_by_h = {}, {}
    if not os.path.exists(path):
        return rows_by_h, err_by_h
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception:
        return rows_by_h, err_by_h
    for h in h_list:
        rs, es = f"rows_h{h}", f"err_h{h}"
        if rs in xl.sheet_names and es in xl.sheet_names:
            rows_by_h[h] = pd.read_excel(xl, sheet_name=rs).to_dict("records")
            err_by_h[h] = pd.read_excel(xl, sheet_name=es)
    return rows_by_h, err_by_h


def _save_checkpoint_h(cc, tag, h, rows_h, err_h):
    path = _ckpt_path(cc, tag)
    os.makedirs("result", exist_ok=True)
    mode = "a" if os.path.exists(path) else "w"
    kw = dict(engine="openpyxl", mode=mode)
    if mode == "a":
        kw["if_sheet_exists"] = "replace"
    with pd.ExcelWriter(path, **kw) as w:
        pd.DataFrame(rows_h).to_excel(w, sheet_name=f"rows_h{h}", index=False)
        err_h.to_excel(w, sheet_name=f"err_h{h}", index=False)


def _rank_block(blk, cols):
    """Rank-score scheme on `cols` (lower = better); non-finite rows excluded."""
    finite = np.isfinite(blk[cols]).all(axis=1)
    if (~finite).any():
        print(f"  ⚠ rank: excluded non-finite: {blk.loc[~finite, 'Model'].tolist()}")
    blk = blk[finite].copy()
    for c in cols:
        r = blk[c].rank(ascending=True, method="min")
        blk[f"{c}_scr"] = (len(blk) + 1 - r).astype(int)
    blk["TOTAL_scr"] = blk[[f"{c}_scr" for c in cols]].sum(axis=1)
    return blk.sort_values("TOTAL_scr", ascending=False)


def run_country(cc, target_col, dataset_path="dataset/ds_steel.xlsx",
                h_list=H_LIST, seeds=SEEDS, n_trials=N_TRIALS,
                models=DL_MODELS, force_tune=False, oil_path=None,
                tag="", resume=True):
    """Probabilistic arm. Exports result/v2p_result_steel_<cc>.xlsx with:
    - summary (point + quantile metrics per model),
    - rank_prob (PB05/PB95/Winkler90, quantile-capable models only),
    - rank (SMAPE/RMSE/MASE, all models — FRI-compatible),
    - error sheets with {pfx}_pred / {pfx}_q05 / {pfx}_q95 for the VaR layer.
    Crash-safety and resume semantics identical to V2 (v2p_ checkpoint files)."""
    print(f"{'='*64}\n{cc}: preprocessing (65/15/20 split)...")
    d = preprocess(target_col, dataset_path, oil_path=oil_path)
    print(f"  n={d['n']} | train until {d['train_o'].end_time().date()} | "
          f"val until {d['val_o'].end_time().date()} | test until {d['test_o'].end_time().date()}")
    print(f"  interp_share={d['interp_share']:.3f} | staleness={d['stale_raw']:.3f}")

    ckpt_rows, ckpt_err = ({}, {}) if not resume else _load_checkpoint(cc, tag, h_list)
    if ckpt_rows:
        print(f"  ↩ resuming: horizons {sorted(ckpt_rows)} loaded from checkpoint")

    all_rows, err_sheets, val_pb = [], {}, {}
    for h in h_list:
        if h in ckpt_rows and h in ckpt_err:
            all_rows.extend(ckpt_rows[h])
            err_sheets[h] = ckpt_err[h]
            print(f"\n----- {cc} | h={h}  (from checkpoint, skipped) -----")
            continue
        print(f"\n----- {cc} | h={h} -----")
        rows_h = []
        test_idx, y_ref, naive_dict, scale_h = naive_runs(d, h)
        pos_ref = {dt: i for i, dt in enumerate(test_idx)}

        preds_test, q_test, seed_disp = {}, {}, {}
        for kind in models:
            hp = load_or_tune(cc, kind, d, h, n_trials=n_trials, force=force_tune)
            val_pb[(kind, h)] = float(hp.get("_val_pinball", np.nan))
            runs = final_runs(kind, hp, d, h, seeds=seeds)
            per_seed = []
            for seed, idx, y, Q in runs:
                sel = np.array([pos_ref[dt] for dt in idx if dt in pos_ref])
                mask = np.isin(idx, test_idx)
                full = {}
                for q in QUANTILES:
                    a = np.full(len(test_idx), np.nan)
                    if len(sel):
                        a[sel] = Q[q][mask]
                    full[q] = a
                v = ~np.isnan(full[0.5])
                if v.any():
                    pb = float(np.mean([pinball(y_ref[v], full[q][v], q)
                                        for q in QUANTILES]))
                    sm = float(np.mean(100*2*np.abs(y_ref[v]-full[0.5][v]) /
                                       (np.abs(y_ref[v])+np.abs(full[0.5][v])+EPS)))
                else:
                    pb, sm = np.nan, np.nan
                per_seed.append((seed, full, pb, sm))
            # median seed by pinball (aligned with the tuning objective)
            per_seed.sort(key=lambda t: (np.isnan(t[2]), t[2]))
            med = per_seed[len(per_seed)//2]
            preds_test[kind] = med[1][0.5]
            q_test[kind] = {"q05": med[1][0.05], "q95": med[1][0.95]}
            seed_disp[kind] = float(np.nanstd([t[2] for t in per_seed]))
            print(f"  [{kind}|h={h}] test pinball={med[2]:.5f} | SMAPE(med)={med[3]:.4f} "
                  f"(±{seed_disp[kind]:.5f} pb across {len(seeds)} seeds)")

        for name, (idx, y, yh) in stat_baseline_runs(d, h).items():
            sel = np.array([pos_ref[dt] for dt in idx if dt in pos_ref])
            yh_full = np.full(len(test_idx), np.nan)
            mask = np.isin(idx, test_idx)
            yh_full[sel] = yh[mask]
            preds_test[name] = yh_full

        # Top-3 frozen on VALIDATION (pinball); all-NaN guard kept from V2 fix
        dl_val = {k: v for (k, hh), v in val_pb.items() if hh == h and k in models}
        dead = [k for k in dl_val if np.isnan(preds_test[k]).all()]
        if dead:
            print(f"  ⚠ [{cc}|h={h}] all-NaN test forecasts, excluded from Top-3: {dead}")
        top3 = sorted((k for k in dl_val if k not in dead), key=dl_val.get)[:3]
        print(f"  Top-3 (validation, pinball): {top3}")

        ok = ~np.isnan(np.column_stack([preds_test[m] for m in top3])).any(axis=1)
        if ok.sum() == 0:
            raise RuntimeError(f"[{cc}|h={h}] top3 {top3} with 100% NaN predictions "
                               f"— horizon NOT checkpointed.")
        y_c = y_ref[ok]
        P_c = {m: preds_test[m][ok] for m in preds_test}
        Q05 = {m: q_test[m]["q05"][ok] for m in top3}
        Q95 = {m: q_test[m]["q95"][ok] for m in top3}
        naive_c = naive_dict["NaiveH"][ok]
        ens, ens_q = ensembles_v2(y_c, {m: P_c[m] for m in top3}, Q05, Q95,
                                  top3, naive_c, h)

        allp = {**{m: P_c[m] for m in preds_test}, **ens,
                "NaiveH": naive_c, "DriftH": naive_dict["DriftH"][ok]}
        allq = {**{m: (q_test[m]["q05"][ok], q_test[m]["q95"][ok]) for m in q_test},
                **ens_q}
        mae_naive = np.mean(np.abs(y_c - naive_c))
        for name, yh in allp.items():
            v = ~np.isnan(yh)
            m = np_metrics(y_c[v], yh[v], scale_h)
            e_m, e_n = y_c[v] - yh[v], (y_c - naive_c)[v]
            st, p1 = dm_overlapping(e_m, e_n, L=h+4)
            sts, ps1 = dm_stride(e_m, e_n, h)
            if name in allq:
                q05a, q95a = allq[name]
                vq = v & ~np.isnan(q05a) & ~np.isnan(q95a)
                m.update(q_metrics(y_c[vq], q05a[vq], q95a[vq]))
            else:
                m.update(PB05=np.nan, PB95=np.nan, Cover90=np.nan, Winkler90=np.nan)
            m.update(Model=name, Dataset=cc, h=h,
                     RelMAE_vs_NaiveH=float(np.mean(np.abs(e_m))/mae_naive),
                     DM_stat=st, DM_p_one=p1, DMstride_p_one=ps1,
                     Seed_std_PB=seed_disp.get(name, np.nan),
                     Val_Pinball=val_pb.get((name, h), np.nan))
            rows_h.append(m)

        # error sheet for the VaR layer: median + native quantiles
        err = pd.DataFrame({"date": np.array(test_idx)[ok],
                            "actual_reconstructed": y_c})
        for name, yh in allp.items():
            pfx = MODEL_PREFIX.get(name, name)
            err[f"{pfx}_pred"] = yh
            err[f"{pfx}_resid"] = yh - y_c
            err[f"{pfx}_error"] = np.abs(yh - y_c)
            if name in allq:
                q05a, q95a = allq[name]
                err[f"{pfx}_q05"] = q05a
                err[f"{pfx}_q95"] = q95a

        _save_checkpoint_h(cc, tag, h, rows_h, err)
        all_rows.extend(rows_h)
        err_sheets[h] = err
        print(f"  ✔ h={h} checkpointed ({_ckpt_path(cc, tag)})")

    summary = pd.DataFrame(all_rows)
    rankp_frames, rankq_frames = [], []
    for h in h_list:
        blk = summary[summary.h == h].copy()
        rankp_frames.append(_rank_block(blk, ["SMAPE", "RMSE", "MASE"]))
        blkq = blk[blk[["PB05", "PB95", "Winkler90"]].notna().all(axis=1)].copy()
        if len(blkq):
            rankq_frames.append(_rank_block(blkq, ["PB05", "PB95", "Winkler90"]))
    rank = pd.concat(rankp_frames, ignore_index=True)
    rank_prob = (pd.concat(rankq_frames, ignore_index=True)
                 if rankq_frames else pd.DataFrame())

    sfx = f"_{tag}" if tag else ""
    out = f"result/v2p_result_steel_{cc.lower()}{sfx}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name=f"summary_{cc.lower()}", index=False)
        rank_prob.to_excel(w, sheet_name="rank_prob", index=False)
        rank.to_excel(w, sheet_name="rank", index=False)
        for h, err in err_sheets.items():
            err.to_excel(w, sheet_name=f"error_{cc.lower()}_h{h}", index=False)
        pd.DataFrame([dict(interp_share=d["interp_share"],
                           staleness=d["stale_raw"], n=d["n"],
                           train_end=str(d["train_o"].end_time().date()),
                           val_end=str(d["val_o"].end_time().date()),
                           test_end=str(d["test_o"].end_time().date()))]
                     ).to_excel(w, sheet_name="data_quality", index=False)
    print(f"\n✅ {out} saved")
    return dict(d=d, summary=summary, rank=rank, rank_prob=rank_prob,
                err_sheets=err_sheets)


# ============================================================
# 9. PLOTS (paper-ready; interval plot added)
# ============================================================
def plot_split_overview(d, cc, figsize=(22, 5)):
    import matplotlib.pyplot as plt
    fig, axx = plt.subplots(figsize=figsize)
    for part, color, lbl in [("train_o", "#1f77b4", "Train (65%)"),
                             ("val_o", "#ff7f0e", "Validation (15%)"),
                             ("test_o", "#2ca02c", "Test (20%)")]:
        s = d[part]
        axx.plot(s.time_index, s.values().flatten(), color=color, lw=1.0, label=lbl)
    axx.set_title(f"{cc} — {d['target_col']} | temporal split 65/15/20")
    axx.legend(); axx.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_relmae_by_h(summary, cc, models=None, figsize=(11, 6)):
    import matplotlib.pyplot as plt
    fig, axx = plt.subplots(figsize=figsize)
    models = models or [m for m in summary.Model.unique() if m not in ("NaiveH",)]
    for m in models:
        blk = summary[summary.Model == m].sort_values("h")
        axx.plot(blk["h"], blk["RelMAE_vs_NaiveH"], marker="o", label=m, alpha=0.8)
    axx.axhline(1.0, color="black", ls="--", lw=1.5, label="Naïve-h (=1)")
    axx.set_xscale("log"); axx.set_xticks(summary["h"].unique())
    axx.get_xaxis().set_major_formatter(__import__("matplotlib").ticker.ScalarFormatter())
    axx.set_xlabel("Horizon h (business days)"); axx.set_ylabel("RelMAE vs. naive-h")
    axx.set_title(f"{cc} — RelMAE by horizon (below 1 = beats the random walk)")
    axx.legend(ncol=2, fontsize=9); axx.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_test_forecast(err_sheet, model_prefix, cc, h, figsize=(22, 6)):
    import matplotlib.pyplot as plt
    fig, axx = plt.subplots(figsize=figsize)
    axx.plot(err_sheet["date"], err_sheet["actual_reconstructed"],
             color="black", ls=":", lw=1.5, label="Observed")
    axx.plot(err_sheet["date"], err_sheet[f"{model_prefix}_pred"],
             color="#1f77b4", lw=1.0, label=model_prefix)
    if f"{model_prefix}_q05" in err_sheet.columns:
        axx.fill_between(err_sheet["date"], err_sheet[f"{model_prefix}_q05"],
                         err_sheet[f"{model_prefix}_q95"], color="#1f77b4",
                         alpha=0.18, label="90% interval (q05–q95)")
    axx.plot(err_sheet["date"], err_sheet["naive_pred"],
             color="#d62728", lw=0.8, alpha=0.6, label="Naïve-h")
    axx.set_title(f"{cc} — test set, h={h}: {model_prefix} vs. naive-h")
    axx.legend(); axx.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_interval_coverage(err_sheet, model_prefix, cc, h, figsize=(22, 5)):
    """Exceedances of the native 90% interval — visual companion of Cover90."""
    import matplotlib.pyplot as plt
    y = err_sheet["actual_reconstructed"].values
    q05 = err_sheet[f"{model_prefix}_q05"].values
    q95 = err_sheet[f"{model_prefix}_q95"].values
    exc_hi = y > q95; exc_lo = y < q05
    fig, axx = plt.subplots(figsize=figsize)
    axx.fill_between(err_sheet["date"], q05, q95, color="#1f77b4", alpha=0.18,
                     label="90% interval")
    axx.plot(err_sheet["date"], y, color="black", lw=0.7, label="Observed")
    axx.scatter(err_sheet["date"][exc_hi], y[exc_hi], s=16, color="#d62728",
                zorder=5, label=f"above q95 ({int(exc_hi.sum())})")
    axx.scatter(err_sheet["date"][exc_lo], y[exc_lo], s=16, color="#ff7f0e",
                zorder=5, label=f"below q05 ({int(exc_lo.sum())})")
    cov = 1 - (exc_hi.mean() + exc_lo.mean())
    axx.set_title(f"{cc} h={h} — {model_prefix}: empirical coverage {cov:.1%} "
                  f"(nominal 90%)")
    axx.legend(); axx.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
