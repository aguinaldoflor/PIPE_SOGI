"""
PIPE-SOGI v2p — VaR/ES layer for the PROBABILISTIC arm (95% CI protocol)
=========================================================================
Derived from var_v2.py (kept intact for the point/baseline arm).

Changes vs. v2 (each tied to the thesis pivot):
1. NEW METHOD "NativeQR": the VaR bound comes DIRECTLY from the network's
   q95 head exported by pipeline_v2p (columns {pfx}_q95 / {pfx}_q05).
   In loss space: V_t = ln(q95_t / median_t)  (> 0 by construction after the
   monotonic rearrangement), exception when L_t = ln(y_t/median_t) > V_t,
   i.e. exactly y_t > q95_t. No rolling estimation, no residual modeling —
   this is the "learned uncertainty" channel competing against the two-stage
   residual methods on the SAME evaluation sample.
   ES_t for NativeQR is the empirical mean of window losses above V_t
   (transparent hybrid, documented: the 3 trained quantiles do not identify
   the tail beyond q95).
2. 95%-ONLY BACKTEST. Single confidence level A = 0.95. Rationale (thesis):
   corporate treasury context (no Basel capital requirement) + statistical
   power: with n≈1000, 95% yields ~50 expected exceedances (Kupiec/
   Christoffersen with real discriminating power) vs ~10 at 99%.
   Kupiec95 acceptance, Christoffersen independence and DQ at 95%.
3. Traffic light: the Basel-style zone classification is recomputed from
   Binomial(n, 0.05) for the actual n ("TL95" — an adaptation, NOT Basel,
   which is defined at 99%; documented as such). k rebased to 1.0 (Green),
   Rejected floor 1.1333 kept.
4. FRI v2p = SMAPE_norm × R_norm × k with R = mean(ES95/VaR95) >= 1.
   WHS lambda calibrated on the first half by Kupiec95 (was Kupiec99).
Input files: result/v2p_result_steel_<cc>.xlsx (pipeline_v2p exports).
Engine selection: read Top-3 from the `rank_prob` sheet (pinball/Winkler),
NOT from the point rank — see the companion notebook.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chi2, binom, genpareto

EPS = 1e-12
A = 0.95                          # single confidence level (v2p protocol)
WINDOW = 252                      # rolling estimation window
WHS_GRID = (0.80, 0.85, 0.90, 0.94, 0.96, 0.98, 0.99)
GARCH_REFIT = 63                  # re-estimate GARCH params quarterly

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


# ------------------------------------------------------------------
# 1. Losses (and native quantile bounds) from v2p error sheets
# ------------------------------------------------------------------
def load_losses(result_file, cc, h, prefix):
    """Signed procurement loss L_t = ln(y_t / median_t). If the engine has
    native quantile columns, also returns the native bounds in loss space:
    Vnat95_t = ln(q95_t / median_t), Vnat05_t = ln(q05_t / median_t)."""
    err = pd.read_excel(result_file, sheet_name=f"error_{cc.lower()}_h{h}")
    y = err["actual_reconstructed"].to_numpy(float)
    yh = err[f"{prefix}_pred"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(yh) & (y > 0) & (yh > 0)
    L = np.log(y[ok] / yh[ok])
    dates = pd.to_datetime(err["date"])[ok].reset_index(drop=True)
    native = None
    if f"{prefix}_q95" in err.columns:
        q95 = err[f"{prefix}_q95"].to_numpy(float)[ok]
        q05 = err[f"{prefix}_q05"].to_numpy(float)[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            native = dict(V95=np.log(np.maximum(q95, EPS) / yh[ok]),
                          V05=np.log(np.maximum(q05, EPS) / yh[ok]))
    return dates, L, native


# ------------------------------------------------------------------
# 2. VaR/ES methods (right tail of losses), rolling one-step-ahead
#    Each returns (VaR_t, ES_t) using ONLY the window ending at t-1.
# ------------------------------------------------------------------
def _hist_es(w, var):
    tail = w[w >= var]
    return float(tail.mean()) if len(tail) else float(var)

def var_hist(w, a):
    v = float(np.quantile(w, a))
    return v, _hist_es(w, v)

def var_whs(w, a, lam):
    n = len(w)
    wt = lam ** np.arange(n - 1, -1, -1)
    wt = wt / wt.sum()
    order = np.argsort(w)
    cs = np.cumsum(wt[order])
    idx = np.searchsorted(cs, a)
    idx = min(idx, n - 1)
    v = float(w[order][idx])
    sel = w >= v
    es = float(np.sum(w[sel] * wt[sel]) / max(wt[sel].sum(), EPS))
    return v, es

def var_cf(w, a):
    mu, sd = w.mean(), w.std(ddof=1)
    S, K = stats.skew(w), stats.kurtosis(w)      # excess kurtosis
    z = norm.ppf(a)
    zcf = (z + (z**2 - 1) * S / 6 + (z**3 - 3*z) * K / 24
           - (2*z**3 - 5*z) * (S**2) / 36)
    v = mu + sd * zcf
    grid = np.linspace(a, 0.9995, 25)
    zs = norm.ppf(grid)
    zcfs = (zs + (zs**2 - 1)*S/6 + (zs**3 - 3*zs)*K/24 - (2*zs**3 - 5*zs)*(S**2)/36)
    es = float(np.mean(mu + sd * zcfs))
    return float(v), es

def var_evt(w, a, u_q=0.90):
    u = np.quantile(w, u_q)
    exc = w[w > u] - u
    if len(exc) < 15:
        return var_hist(w, a)
    try:
        xi, loc, beta = genpareto.fit(exc, floc=0.0)
    except Exception:
        return var_hist(w, a)
    n, nu = len(w), len(exc)
    p_exc = nu / n
    if (1 - a) >= p_exc:
        return var_hist(w, a)
    v = u + (beta / xi) * (((1 - a) / p_exc) ** (-xi) - 1) if abs(xi) > 1e-9 \
        else u + beta * np.log(p_exc / (1 - a))
    if xi < 1:
        es = (v / (1 - xi)) + (beta - xi * u) / (1 - xi) if abs(xi) > 1e-9 else v + beta
    else:
        es = v * 1.5
    return float(v), float(max(es, v))


def _ewma_sigma(L, lam=0.94):
    s2 = np.empty(len(L))
    s2[0] = np.var(L[:20]) if len(L) >= 20 else np.var(L) + EPS
    for t in range(1, len(L)):
        s2[t] = lam * s2[t-1] + (1 - lam) * L[t-1]**2
    return np.sqrt(np.maximum(s2, EPS))


def rolling_var(L, method, a, lam_whs=0.94):
    """Returns arrays VaR[t], ES[t] for t = WINDOW..n-1 (forecast for step t)."""
    n = len(L)
    V = np.full(n, np.nan); E = np.full(n, np.nan)
    if method == "FHS-EWMA":
        sig = _ewma_sigma(L, 0.94)
        z = L / np.maximum(sig, EPS)
        for t in range(WINDOW, n):
            zw = z[t-WINDOW:t]
            qz = np.quantile(zw, a)
            tail = zw[zw >= qz]
            V[t] = sig[t] * qz
            E[t] = sig[t] * (tail.mean() if len(tail) else qz)
        return V, E
    if method == "CornishFisher":
        s = pd.Series(L)
        mu = s.rolling(WINDOW).mean().shift(1).to_numpy()
        sd = s.rolling(WINDOW).std(ddof=1).shift(1).to_numpy()
        sk = s.rolling(WINDOW).skew().shift(1).to_numpy()
        ku = s.rolling(WINDOW).kurt().shift(1).to_numpy()   # excess kurtosis
        z = norm.ppf(a)
        grid = norm.ppf(np.linspace(a, 0.9995, 25))
        for t in range(WINDOW, n):
            S, K = sk[t], ku[t]
            zcf = (z + (z**2 - 1)*S/6 + (z**3 - 3*z)*K/24 - (2*z**3 - 5*z)*(S**2)/36)
            V[t] = mu[t] + sd[t] * zcf
            zcfs = (grid + (grid**2 - 1)*S/6 + (grid**3 - 3*grid)*K/24
                    - (2*grid**3 - 5*grid)*(S**2)/36)
            E[t] = mu[t] + sd[t] * float(np.mean(zcfs))
        return V, E
    if method == "EVT":
        vv_, ee_ = np.nan, np.nan
        for t in range(WINDOW, n):
            if (t - WINDOW) % 5 == 0:
                vv_, ee_ = var_evt(L[t-WINDOW:t], a)
            V[t], E[t] = vv_, ee_
        return V, E
    for t in range(WINDOW, n):
        w = L[t-WINDOW:t]
        if method == "Historical":
            V[t], E[t] = var_hist(w, a)
        elif method == "WeightedHS":
            V[t], E[t] = var_whs(w, a, lam_whs)
    return V, E


def rolling_var_garch(L, a, kind="GJR"):
    """GJR-GARCH-t / EGARCH-skewt: params re-estimated every GARCH_REFIT obs on
    the trailing window (<=750), variance recursion with FIXED params in between
    (the statistical analogue of retrain=False)."""
    if not HAS_ARCH:
        return rolling_var(L, "FHS-EWMA", a)
    n = len(L)
    V = np.full(n, np.nan); E = np.full(n, np.nan)
    x = L * 100.0                                  # arch scaling
    res = None
    dist = "t" if kind == "GJR" else "skewt"
    sig2 = np.var(x[:WINDOW])
    for t in range(WINDOW, n):
        if res is None or (t - WINDOW) % GARCH_REFIT == 0:
            lo = max(0, t - 750)
            try:
                am = (arch_model(x[lo:t], p=1, o=1, q=1, dist=dist)
                      if kind == "GJR" else
                      arch_model(x[lo:t], vol="EGARCH", p=1, o=1, q=1, dist=dist))
                res = am.fit(disp="off", show_warning=False)
                fc = res.forecast(horizon=1, reindex=False)
                sig2 = float(fc.variance.values[-1, 0])
                params = res.params
            except Exception:
                res = None
        if res is None:
            V[t], E[t] = var_hist(x[t-WINDOW:t] / 100.0, a)
            continue
        try:
            if dist == "t":
                nu = float(params.get("nu", 8.0))
                q = stats.t.ppf(a, nu) * np.sqrt((nu - 2) / nu)
                zs = np.linspace(a, 0.9995, 25)
                qE = np.mean(stats.t.ppf(zs, nu)) * np.sqrt((nu - 2) / nu)
            else:
                from arch.univariate import SkewStudent
                nu = float(params.get("eta", params.get("nu", 8.0)))
                lb = float(params.get("lambda", 0.0))
                sk = SkewStudent()
                q = float(sk.ppf(a, [nu, lb]))
                zs = np.linspace(a, 0.9995, 25)
                qE = float(np.mean(sk.ppf(zs, [nu, lb])))
        except Exception:
            q = norm.ppf(a); qE = float(np.mean(norm.ppf(np.linspace(a, 0.9995, 25))))
        mu = float(params.get("mu", 0.0))
        if not np.isfinite(sig2) or sig2 < 1e-8 or sig2 > 1e6:
            sig2 = float(np.var(x[max(0, t-WINDOW):t]))   # reset defensivo
        s = np.sqrt(max(sig2, EPS))
        V[t] = (mu + s * q) / 100.0
        E[t] = (mu + s * qE) / 100.0
        e = x[t] - mu
        if kind == "GJR":
            om = float(params.get("omega", 0.0)); al = float(params.get("alpha[1]", 0.05))
            ga = float(params.get("gamma[1]", 0.0)); be = float(params.get("beta[1]", 0.9))
            sig2 = om + (al + ga * (e < 0)) * e**2 + be * sig2
        else:
            om = float(params.get("omega", 0.0)); al = float(params.get("alpha[1]", 0.1))
            ga = float(params.get("gamma[1]", 0.0)); be = float(params.get("beta[1]", 0.95))
            z = e / max(np.sqrt(sig2), EPS)
            Eabs = np.sqrt(2 / np.pi)
            lg = om + be * np.log(max(sig2, EPS)) + al * (abs(z) - Eabs) + ga * z
            sig2 = float(np.exp(np.clip(lg, -30.0, 15.0)))   # anti-overflow
    return V, E


METHODS = ["Historical", "WeightedHS", "CornishFisher", "FHS-EWMA", "EVT",
           "GJR-GARCH-t", "EGARCH-skewt"]

def compute_var(L, method, a, lam_whs=0.94):
    if method == "GJR-GARCH-t":
        return rolling_var_garch(L, a, "GJR")
    if method == "EGARCH-skewt":
        return rolling_var_garch(L, a, "EGARCH")
    return rolling_var(L, method, a, lam_whs)


def conformal_qr(L, Vnat, window=WINDOW, alpha=1.0 - A, vol_scale=True):
    """Rolling normalized split-conformal correction of the network's native VaR
    (Romano et al. 2019, CQR + locally-adaptive/normalized conformal).

    Fixes the two failures of raw NativeQR:
      (1) LEVEL — at each t, calibrate on the trailing `window` so the empirical
          exceedance rate matches 1-A (finite-sample corrected quantile). A band
          that is too wide (conservative, e.g. CHI) is tightened; too narrow is
          widened.
      (2) CLUSTERING — conformity scores are normalized by a local volatility
          proxy sigma_t (EWMA of L); the correction is rescaled by sigma_t, so
          the band BREATHES with volatility. This is what lets it react to the
          regime the static native band ignores (attacks Christoffersen).

    Rolling (not validation-calibrated) by design: same 252-day protocol as the
    other VaR methods, so the comparison is fair and no pipeline rerun is needed.
    Set vol_scale=False to recover plain CQR (level fix only, no reactivity)."""
    n = len(L)
    V = np.full(n, np.nan); E = np.full(n, np.nan)
    sig = _ewma_sigma(L, 0.94) if vol_scale else np.ones(n)
    for t in range(window, n):
        if not np.isfinite(Vnat[t]):
            continue
        base = Vnat[t-window:t]; Lw = L[t-window:t]
        sw = np.maximum(sig[t-window:t], EPS)
        s = (Lw - base) / sw
        s = s[np.isfinite(s)]
        if len(s) < 20:
            V[t] = Vnat[t]; continue
        k = int(np.ceil((len(s) + 1) * (1.0 - alpha)))
        k = min(max(k, 1), len(s))
        q = np.sort(s)[k-1]                       # finite-sample conformal quantile
        V[t] = Vnat[t] + sig[t] * q               # level-corrected + vol-reactive
        tail = Lw[Lw > V[t]]
        E[t] = tail.mean() if len(tail) else V[t]
    return V, E


ACI_GAMMA = 0.05   # pre-specified step size; validation-calibration = future work


def aci_qr(L, Vnat, window=WINDOW, atgt=1.0 - A, gamma=ACI_GAMMA):
    """Adaptive Conformal Inference (Gibbs & Candès 2021) over the network's
    native VaR, volatility-normalized. The effective miscoverage level a_t is
    updated ONLINE from realized breaches:
        a_{t+1} = clip(a_t + gamma * (atgt - err_t), 0.001, 0.5),
    with err_t = 1{L_t > VaR_t}. After a breach the level rises -> the band
    widens; after quiet spells it tightens. This targets CONDITIONAL coverage
    directly (the dynamic-quantile / clustering failure that the static
    ConformalQR does not fully fix).

    gamma is fixed at ACI_GAMMA (pre-specified). Calibrating gamma on the
    validation set — which requires exporting validation-period native
    quantiles from the pipeline — is left as future work."""
    n = len(L)
    V = np.full(n, np.nan); E = np.full(n, np.nan)
    sig = _ewma_sigma(L, 0.94)
    a_t = atgt
    for t in range(window, n):
        if not np.isfinite(Vnat[t]):
            continue
        base = Vnat[t-window:t]; Lw = L[t-window:t]
        sw = np.maximum(sig[t-window:t], EPS)
        s = (Lw - base) / sw
        s = s[np.isfinite(s)]
        if len(s) < 20:
            V[t] = Vnat[t]; continue
        lvl = min(max(1.0 - a_t, 0.5), 0.999)
        k = int(np.ceil((len(s) + 1) * lvl)); k = min(max(k, 1), len(s))
        V[t] = Vnat[t] + sig[t] * np.sort(s)[k-1]
        tail = Lw[Lw > V[t]]
        E[t] = tail.mean() if len(tail) else V[t]
        err = 1.0 if L[t] > V[t] else 0.0
        a_t = min(max(a_t + gamma * (atgt - err), 0.001), 0.5)
    return V, E


def ensemble_fz0(members, y, window=WINDOW, lam=50.0, tau=A):
    """RISK ensemble (Opção 1): combina previsões de risco no ESPAÇO DE PREÇO
    com pesos softmax pela perda FZ0 recente — não por média de quantis, e de
    forma COERENTE (cada modelo tem seu próprio ponto, então combina-se em preço
    absoluto e avalia-se contra uma referência comum).

    members : dict {nome: (yhat, q95, es)} — séries de PREÇO por modelo:
              yhat = ponto (mediana), q95 = nível de VaR 95% (preço),
              es = nível de ES 95% (preço), alinhadas ao preço realizado y.
    lam     : temperatura do softmax (maior = concentra no melhor membro).

    Em cada t: peso w_i ∝ exp(-lam * FZ0_i médio na janela), onde FZ0_i é medido
    no espaço de perda do PRÓPRIO membro; combina em preço (q95_ens=Σ w_i q95_i,
    es_ens, ponto m_ens); devolve (L, V, E) no espaço de perda relativo a m_ens:
        L = ln(y/m_ens),  V = ln(q95_ens/m_ens),  E = ln(es_ens/m_ens).
    Coerente: y é comum e tudo é medido relativo ao ponto do ensemble.
    Não re-treina nada — opera sobre séries já produzidas. Retorna (L, V, E)."""
    names = list(members)
    YH = np.column_stack([np.asarray(members[n][0], float) for n in names])   # ponto
    Q = np.column_stack([np.asarray(members[n][1], float) for n in names])    # q95 preço
    S = np.column_stack([np.asarray(members[n][2], float) for n in names])    # es preço
    y = np.asarray(y, float); T, M = YH.shape

    # FZ0 de cada membro no seu próprio espaço de perda (base dos pesos)
    fz = np.column_stack([
        fz0_loss(np.log(y / YH[:, j]),
                 np.maximum(np.log(np.maximum(Q[:, j], 1e-9) / YH[:, j]), 1e-4),
                 np.log(np.maximum(S[:, j], 1e-9) / YH[:, j]), tau)
        for j in range(M)])

    L = np.full(T, np.nan); V = np.full(T, np.nan); E = np.full(T, np.nan)
    for t in range(window, T):
        score = np.nanmean(fz[t - window:t], axis=0)
        if not np.isfinite(score).any():
            continue
        w = np.exp(-lam * (score - np.nanmin(score))); w = w / w.sum()
        m = float(np.dot(w, YH[t])); q = float(np.dot(w, Q[t])); s = float(np.dot(w, S[t]))
        L[t] = np.log(y[t] / m)
        V[t] = np.log(max(q, 1e-9) / m)
        E[t] = max(np.log(max(s, 1e-9) / m), V[t])
    return L, V, E


def native_var(L, Vnat):
    """NativeQR: V_t comes straight from the network's q95 head (loss space).
    ES_t = empirical mean of window losses above V_t (documented hybrid: the
    trained quantiles do not identify the tail beyond q95). Same WINDOW warmup
    as the rolling methods, so every method is backtested on the SAME sample."""
    n = len(L)
    V = np.asarray(Vnat, float).copy()
    E = np.full(n, np.nan)
    for t in range(WINDOW, n):
        if not np.isfinite(V[t]):
            continue
        w = L[t-WINDOW:t]
        tail = w[w > V[t]]
        E[t] = tail.mean() if len(tail) else V[t]
    V[:WINDOW] = np.nan
    return V, E


# ------------------------------------------------------------------
# 3. Backtests (single level A = 0.95)
# ------------------------------------------------------------------
def kupiec(hits, a):
    n, x = len(hits), int(hits.sum())
    p = 1 - a
    if n == 0:
        return np.nan
    pi = max(x / n, EPS)
    lr = -2 * ((n - x) * np.log(max(1 - p, EPS)) + x * np.log(p)
               - (n - x) * np.log(max(1 - pi, EPS)) - x * np.log(pi))
    return float(1 - chi2.cdf(lr, 1))

def christoffersen(hits):
    h = hits.astype(int)
    if h.sum() < 2:
        return np.nan
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(h)):
        if h[i-1] == 0 and h[i] == 0: n00 += 1
        elif h[i-1] == 0 and h[i] == 1: n01 += 1
        elif h[i-1] == 1 and h[i] == 0: n10 += 1
        else: n11 += 1
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def sll(p, a_, b_): return a_ * np.log(max(1 - p, EPS)) + b_ * np.log(max(p, EPS))
    lr = -2 * (sll(p1, n00 + n10, n01 + n11)
               - sll(p01, n00, n01) - sll(p11, n10, n11))
    return float(1 - chi2.cdf(lr, 1))

def christoffersen_stride(hits, h):
    """Christoffersen independence on DE-OVERLAPPED exceptions (every h-th obs).
    At h>1 the one-shot forecasts overlap, inducing MECHANICAL dependence in the
    hit sequence that contaminates the standard test; sampling every h-th point
    removes it. Median over the h possible phase offsets. Same rationale as the
    stride-h Diebold-Mariano test used in the forecasting layer."""
    if h <= 1:
        return christoffersen(hits)
    ps = []
    for off in range(h):
        p = christoffersen(hits[off::h])
        if np.isfinite(p):
            ps.append(p)
    return float(np.median(ps)) if ps else np.nan


def dq_test(hits, var_series, a, n_lags=4):
    p = 1 - a
    d = hits.astype(float) - p
    n = len(d)
    if n <= n_lags + 3 or hits.sum() == 0:
        return np.nan
    X = [np.ones(n - n_lags)]
    for l in range(1, n_lags + 1):
        X.append(d[n_lags - l:n - l])
    X.append(var_series[n_lags:])
    X = np.column_stack(X)
    yv = d[n_lags:]
    try:
        b, *_ = np.linalg.lstsq(X, yv, rcond=None)
        stat = float(b @ X.T @ X @ b) / (p * (1 - p))
        return float(1 - chi2.cdf(stat, X.shape[1]))
    except Exception:
        return np.nan


def pinball_loss(realized, var, tau=A):
    """Quantile (pinball) score of a VaR forecast at level tau: proper scoring
    rule, minimized in expectation when `var` is the true tau-quantile of the
    loss. QS95 = mean_t rho_tau(L_t - VaR_t), rho_tau(u)=u(tau-1{u<0}).
    Jointly penalizes under- and over-coverage (calibration) and width
    (sharpness); consistent with the Kupiec95 backtest (same tau)."""
    u = np.asarray(realized, float) - np.asarray(var, float)
    pb = np.where(u >= 0, tau * u, (tau - 1.0) * u)
    return float(np.nanmean(pb))


def dq_stride(hits, var_series, h, n_lags=2):
    """DQ test on DE-OVERLAPPED exceptions (every h-th obs), median over the h
    phase offsets. Same overlap rationale as christoffersen_stride: at h>1 the
    standard DQ is contaminated by the mechanical dependence of overlapping
    one-shot forecasts. Primary dynamic-quantile test at h=5; the full-sample
    DQ95 is reported alongside for transparency."""
    if h <= 1:
        return dq_test(hits, var_series, A, n_lags=n_lags)
    ps = []
    for off in range(h):
        p = dq_test(hits[off::h], var_series[off::h], A, n_lags=n_lags)
        if np.isfinite(p):
            ps.append(p)
    return float(np.median(ps)) if ps else np.nan


def fz0_loss(L, V, E, tau=A):
    """0-homogeneous Fissler-Ziegel loss (Nolde & Ziegel 2017; Patton, Ziegel &
    Chen 2019) — the strictly consistent scoring function for the JOINT
    functional (VaR, ES) at level tau. Upper-tail (procurement-loss) convention,
    v = VaR > 0, e = ES >= v > 0, realized loss L:

        S = 1{L>=v}(L-v) / ((1-tau) e) + v/e + ln(e) - 1

    Lower S = better. Strictly consistent (minimized in expectation by the true
    (VaR, ES)) and scale-invariant (0-homogeneous) — ideal for cross-market
    ranking. Returns the per-observation loss array; the mean is the score."""
    l = np.asarray(L, float); v = np.asarray(V, float); e = np.asarray(E, float)
    e = np.maximum(e, v)                       # enforce ES >= VaR
    v = np.maximum(v, 1e-6); e = np.maximum(e, 1e-6)   # positivity for ln/division
    ind = (l >= v).astype(float)
    return ind * (l - v) / ((1.0 - tau) * e) + v / e + np.log(e) - 1.0


def dm_loss(loss_a, loss_b, nw_lag):
    """Diebold-Mariano comparative test on a loss differential d = loss_a - loss_b
    with Newey-West HAC variance (nw_lag lags, ~h for overlapping forecasts).
    Returns (stat, p_one) where p_one = P(model A has significantly LOWER loss),
    i.e. small p_one => A is significantly better than B."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    dbar = d.mean(); dc = d - dbar
    s = np.mean(dc * dc)
    for l in range(1, nw_lag + 1):
        s += 2 * (1 - l / (nw_lag + 1)) * np.mean(dc[l:] * dc[:-l])
    if s <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(s / T)
    return float(stat), float(norm.cdf(stat))


def traffic_light_95(n, exceptions):
    """Zone classification from Binomial(n, 0.05) for the actual n.
    ADAPTATION of the Basel logic to the 95% protocol (Basel itself is defined
    at 99%; this is documented as a descriptive zone test, not regulation).
    k rebased to 1.0 (Green)."""
    e_g = int(binom.ppf(0.95, n, 0.05))
    e_y = int(binom.ppf(0.9999, n, 0.05))
    if exceptions <= e_g:
        zone, k = "Green", 3.00
    elif exceptions <= e_y:
        zone = "Yellow"
        k = 3.40 + 0.45 * (exceptions - e_g) / max(e_y - e_g, 1)
    else:
        zone, k = "Red", 4.00
    return zone, k / 3.00, e_g, e_y


# ------------------------------------------------------------------
# 4. Full pipeline for one (market, h, engine)
# ------------------------------------------------------------------
def run_engine(result_file, cc, h, prefix, lam_whs_cal=True):
    """Backtests the 7 residual methods + NativeQR (when the engine exports
    native quantiles) at A = 0.95, all on the same evaluation sample."""
    dates, L, native = load_losses(result_file, cc, h, prefix)
    rows, series_store = [], {}
    methods = list(METHODS) + (["NativeQR", "ConformalQR", "ACI-QR"]
                               if native is not None else [])
    for method in methods:
        lam = 0.94
        if method == "WeightedHS" and lam_whs_cal:
            # calibrate lambda on the FIRST HALF only, Kupiec95 max (v2p)
            half = WINDOW + (len(L) - WINDOW) // 2
            best_lam, best_p = 0.94, -1
            for lm in WHS_GRID:
                V, _ = rolling_var(L[:half], "WeightedHS", A, lm)
                hits = (L[:half] > V)[WINDOW:half]
                pk = kupiec(hits, A)
                if np.isfinite(pk) and pk > best_p:
                    best_p, best_lam = pk, lm
            lam = best_lam
        if method == "NativeQR":
            V, E = native_var(L, native["V95"])
        elif method == "ConformalQR":
            V, E = conformal_qr(L, native["V95"])
        elif method == "ACI-QR":
            V, E = aci_qr(L, native["V95"])
        else:
            V, E = compute_var(L, method, A, lam)
        hits = (L > V)[WINDOW:]
        k95 = kupiec(hits, A)
        rejected = not (np.isfinite(k95) and k95 >= 0.05)
        ind95 = christoffersen(hits)
        ind95_s = christoffersen_stride(hits, h)   # de-overlapped (h=5 protocol)
        dq95 = dq_test(hits, V[WINDOW:], A)
        dq95_s = dq_stride(hits, V[WINDOW:], h)     # de-overlapped (primary at h=5)
        zone, k_mult, e_g, e_y = traffic_light_95(len(hits), int(hits.sum()))
        if rejected and k_mult < 1.1333:
            k_mult = 1.1333
            zone = zone + "(floored)"
        # tail severity: mean of daily ES/VaR at 95%, both clamped positive
        V95 = np.maximum(V[WINDOW:], 1e-4)             # economic floor: 0.01% loss
        E95 = np.maximum(E[WINDOW:], V95)              # enforce ES >= VaR
        R = float(np.nanmean(E95 / V95))
        if not np.isfinite(R) or R > 5.0:
            R = np.nan                                  # flag degenerate method
        # QS95: pinball score of the actual VaR series (unfloored) vs realized L
        qs95 = pinball_loss(L[WINDOW:], V[WINDOW:], A)
        # FZ0: joint (VaR, ES) proper score — the primary ranking metric
        fz_arr = fz0_loss(L[WINDOW:], V95, E95, A)
        fz0 = float(np.nanmean(fz_arr))
        rows.append(dict(Market=cc, h=h, Engine=prefix, Method=method,
                         lam=lam if method == "WeightedHS" else np.nan,
                         n=len(hits), exc95=int(hits.sum()),
                         exp95=round(0.05 * len(hits), 1),
                         Kupiec95=round(k95, 4) if np.isfinite(k95) else np.nan,
                         Indep95=round(ind95, 4) if np.isfinite(ind95) else np.nan,
                         Indep95_str=round(ind95_s, 4) if np.isfinite(ind95_s) else np.nan,
                         DQ95=round(dq95, 4) if np.isfinite(dq95) else np.nan,
                         DQ95_str=round(dq95_s, 4) if np.isfinite(dq95_s) else np.nan,
                         Rejected=rejected, TL95=zone, k=round(k_mult, 4),
                         FZ0=round(fz0, 6) if np.isfinite(fz0) else np.nan,
                         QS95=round(qs95, 6) if np.isfinite(qs95) else np.nan,
                         ES_VaR_ratio=round(R, 4) if np.isfinite(R) else np.nan,
                         meanVaR95=round(float(np.nanmean(V95)), 5)))
        series_store[method] = {A: dict(V=V[WINDOW:], E=E[WINDOW:], hits=hits),
                                "fz": fz_arr}
    return pd.DataFrame(rows), series_store, dates[WINDOW:], L[WINDOW:]


def select_best_method(df_engine):
    """Two-step evaluation (Nolde & Ziegel 2017): (1) VALIDITY GATE — keep only
    methods that pass the Kupiec95 traffic light (not Rejected); (2) among those,
    pick the SMALLEST FZ0 (joint VaR+ES proper score). Falls back to all methods
    if none is valid. FZ0 replaces QS95 as the ranking metric: it scores VaR and
    ES jointly and does not mechanically penalize sharpness (the QS95-vs-R
    conflict of the earlier index)."""
    sane = df_engine[df_engine.FZ0.notna()]
    if not len(sane):
        sane = df_engine
    ok = sane[~sane.Rejected]
    pool = ok if len(ok) else sane
    return pool.sort_values("FZ0", ascending=True).iloc[0]


# ------------------------------------------------------------------
# 5. FRI v2p (per market: engines compared at the SAME h, 95% level)
# ------------------------------------------------------------------
def fri_table(best_rows, smape_map):
    """best_rows: list of selected-method rows (one per engine) for one market/h.
    smape_map: {prefix: test SMAPE} — REPORTED for reference only, NOT in the index.

    FRI v3 — two reported quantities, both on the risk scale:

      FZ0        : mean FZ0 loss (strictly consistent joint VaR+ES score).
                   Primary SCORE; LOWER = better. As a 0-homogeneous scoring
                   rule its level is defined only up to an additive constant
                   (dominated by the common ln(e) offset), so its absolute value
                   is not interpretable — only differences between models are.

      FRI_gain   : skill vs. the no-model benchmark, sign-flipped to be intuitive
                       FRI_gain = mean_FZ0(NaiveH) - mean_FZ0(engine)
                   POSITIVE => the engine BEATS the random walk (lower FZ0);
                   0 => ties naive; negative => worse. Higher = better. This is
                   the communicable number; pair it with the DM test (dm_loss)
                   for significance.

    NOTE: no ratio index. FZ0/FZ0_naive is meaningless (ratio of negatives,
    inverts the ranking) and was removed. Theory-grounded (Fissler-Ziegel),
    scale-invariant, no SMAPE, no multiplicative weighting, no R term. Validity
    gated upstream (select_best_method). Sorted by FZ0 ascending (best first)."""
    df = pd.DataFrame(best_rows).copy()
    df["SMAPE"] = df.Engine.map(smape_map)          # reference column only
    base = df[df.Engine == "naive"]
    fz_naive = float(base.FZ0.iloc[0]) if len(base) else float(df.FZ0.median())
    df["FRI_gain"] = fz_naive - df.FZ0              # > 0 => beats naive (higher better)
    # QS95_norm: DIAGNOSTIC only (pinball/VaR sharpness relative to market median);
    # NOT part of the FRI. Lower = sharper VaR. Reported to read the pinball
    # performance alongside the joint FZ0 score.
    med_q = df.QS95.median()
    df["QS95_norm"] = df.QS95 / (med_q if abs(med_q) > EPS else EPS)
    return df.sort_values("FZ0")                     # lower FZ0 = better
