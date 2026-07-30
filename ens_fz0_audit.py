import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import var_v2p as V

A = V.A                 # 0.95
WINDOW = V.WINDOW       # 252
H = 5

RESULT_DIR = "result"
FILES = {                                    # <-- CHECK THESE PATHS
    "CHI": "v2p_result_steel_chi.xlsx",
    "USA": "v2p_result_steel_usa.xlsx",
    "GER": "v2p_result_steel_ger.xlsx",
    "TUR": "v2p_result_steel_tur.xlsx",
}

# Top-3 DL pools, ranked by validation pinball (pipeline_v2p.py, L709-713).
# These are the SAME members the EnsMean/EnsDynSMAPE/etc. rows are built from.
POOLS = {
    "CHI": ["dlinear", "tcn", "tsmixer"],
    "USA": ["dlinear", "tcn", "nhits"],
    "GER": ["dlinear", "tcn", "nhits"],
    "TUR": ["tcn", "dlinear", "tsmixer"],
}


# ---------------------------------------------------------------------------
# 1. Rebuild each member's (median, VaR, ES) in PRICE space, on a common index
# ---------------------------------------------------------------------------
def member_series(path, cc, prefix):
    """Runs the full backtest for one engine, keeps the FZ0-best admissible
    method, and returns a DataFrame indexed by date with the three price-space
    series ensemble_fz0() expects, plus the realized price."""
    df, store, dates, L = V.run_engine(path, cc, H, prefix)
    best = V.select_best_method(df)
    s = store[best.Method][A]

    err = pd.read_excel(path, sheet_name=f"error_{cc.lower()}_h{H}")
    y = err["actual_reconstructed"].to_numpy(float)
    yh = err[f"{prefix}_pred"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(yh) & (y > 0) & (yh > 0)
    y, yh = y[ok][WINDOW:], yh[ok][WINDOW:]

    out = pd.DataFrame(
        {"y": y, "yhat": yh,
         "q95": yh * np.exp(s["V"]),           # VaR back to price space
         "es":  yh * np.exp(np.maximum(s["E"], s["V"]))},
        index=pd.DatetimeIndex(dates),
    )
    return out, best.Method


def build_members(cc):
    """Inner-joins the pool members on date so every scheme is scored on
    exactly the same sample (members can differ in their NaN masks)."""
    path = os.path.join(RESULT_DIR, FILES[cc])
    parts, methods = {}, {}
    for p in POOLS[cc]:
        parts[p], methods[p] = member_series(path, cc, p)
    idx = parts[POOLS[cc][0]].index
    for p in POOLS[cc][1:]:
        idx = idx.intersection(parts[p].index)
    parts = {p: d.loc[idx] for p, d in parts.items()}
    y = parts[POOLS[cc][0]]["y"].to_numpy(float)
    members = {p: (d["yhat"].to_numpy(float),
                   d["q95"].to_numpy(float),
                   d["es"].to_numpy(float)) for p, d in parts.items()}
    return members, y, idx, methods


# ---------------------------------------------------------------------------
# 2. The three combination schemes, all returning (L, V, E) in loss space
# ---------------------------------------------------------------------------
def _to_loss(y, m, q, e):
    L = np.log(y / m)
    Vv = np.log(np.maximum(q, 1e-9) / m)
    Ee = np.maximum(np.log(np.maximum(e, 1e-9) / m), Vv)
    return L, Vv, Ee


def scheme_qavg(members, y):
    """(i) EnsMean — equal-weight pooling of the members' quantiles."""
    M = np.column_stack([members[p][0] for p in members]).mean(1)
    Q = np.column_stack([members[p][1] for p in members]).mean(1)
    S = np.column_stack([members[p][2] for p in members]).mean(1)
    return _to_loss(y, M, Q, S)


def scheme_fz0(members, y, lam=50.0):
    """(ii) EnsFZ0 — softmax weights on recent FZ0 (var_v2p.ensemble_fz0)."""
    return V.ensemble_fz0(members, y, window=WINDOW, lam=lam, tau=A)


def scheme_fz0_optimal(members, y, train_frac=0.5):
    """(iii) FZ0-optimal — simplex weights minimizing FZ0 IN-SAMPLE on the first
    half, then held fixed and applied out-of-sample on the second half. The
    out-of-sample half is what gets scored; this is the combination-puzzle test."""
    names = list(members)
    M = np.column_stack([members[p][0] for p in names])
    Q = np.column_stack([members[p][1] for p in names])
    S = np.column_stack([members[p][2] for p in names])
    T, k = M.shape
    cut = int(T * train_frac)

    def obj(w, sl):
        w = np.abs(w); w = w / max(w.sum(), 1e-12)
        m = M[sl] @ w; q = Q[sl] @ w; s = S[sl] @ w
        L, Vv, Ee = _to_loss(y[sl], m, q, s)
        return float(np.nanmean(V.fz0_loss(L, np.maximum(Vv, 1e-6), Ee, A)))

    w0 = np.full(k, 1.0 / k)
    res = minimize(obj, w0, args=(slice(0, cut),), method="SLSQP",
                   bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = np.abs(res.x); w = w / w.sum()

    L = np.full(T, np.nan); Vv = np.full(T, np.nan); Ee = np.full(T, np.nan)
    sl = slice(cut, T)                                   # OUT-OF-SAMPLE ONLY
    L[sl], Vv[sl], Ee[sl] = _to_loss(y[sl], M[sl] @ w, Q[sl] @ w, S[sl] @ w)
    return L, Vv, Ee, dict(zip(names, w.round(4)))


# ---------------------------------------------------------------------------
# 3. Score a (L, V, E) triple the same way run_engine() does
# ---------------------------------------------------------------------------
def score(L, Vv, Ee, label, cc, mask):
    """Scores on a COMMON evaluation window shared by every scheme.

    This is not cosmetic. FZ0 is a MEAN loss, so a scheme evaluated over a
    different stretch of the sample is not comparable: the difference may be
    the volatility regime of its window rather than the quality of the model.
    Equal-weight pooling is defined from t=0, the FZ0-weighted ensemble needs
    252 days of burn-in, and the learned-weight combination only reports its
    out-of-sample half. Scoring them on their own natural windows compares
    three different periods and answers nothing."""
    L = np.asarray(L, float)[mask]
    Vv = np.maximum(np.asarray(Vv, float)[mask], 1e-4)
    Ee = np.maximum(np.asarray(Ee, float)[mask], Vv)
    hits = L > Vv
    fz = V.fz0_loss(L, Vv, Ee, A)
    R = float(np.nanmean(Ee / Vv))
    return dict(Market=cc, Scheme=label, n=len(L),
                exc95=int(hits.sum()), exp95=round(0.05 * len(L), 1),
                cov95=round(1 - hits.mean(), 4),
                Kupiec95=round(V.kupiec(hits, A), 4),
                FZ0=round(float(np.nanmean(fz)), 4),
                ES_VaR=round(R, 3),
                # run_engine() flags R > 5 as a degenerate tail announcement;
                # such a row must not be reported without investigation.
                degenerate=bool(R > 5.0),
                meanVaR95=round(float(np.nanmean(Vv)), 5)), fz


# ---------------------------------------------------------------------------
def main():
    rows, fzs = [], {}
    for cc in FILES:
        print(f"\n=== {cc} ===")
        members, y, idx, methods = build_members(cc)
        print("  members:", {p: methods[p] for p in members})

        # --- compute all three schemes on the full array first -------------
        trip = {}
        trip["EnsMean (quantile avg.)"] = scheme_qavg(members, y)
        trip["EnsFZ0 (FZ0-weighted)"] = scheme_fz0(members, y)
        Lo, Vo, Eo, w = scheme_fz0_optimal(members, y)
        trip["FZ0-optimal (learned w)"] = (Lo, Vo, Eo)

        # --- COMMON evaluation window: where all three are defined ---------
        mask = np.ones(len(y), bool)
        for L_, V_, E_ in trip.values():
            mask &= np.isfinite(L_) & np.isfinite(V_) & np.isfinite(E_)
        print(f"  common evaluation window: {mask.sum()} of {len(y)} obs"
              f"  ({idx[mask][0].date()} to {idx[mask][-1].date()})")

        for label, (L_, V_, E_) in trip.items():
            r, fz = score(L_, V_, E_, label, cc, mask)
            if label.startswith("FZ0-optimal"):
                r["weights"] = str(w)
            rows.append(r); fzs[(cc, label)] = fz
            flag = "  <-- DEGENERATE R>5" if r["degenerate"] else ""
            print(f"  {label:26s} FZ0={r['FZ0']:+.4f}  cov={r['cov95']:.3f}"
                  f"  R={r['ES_VaR']:.2f}{flag}")

        # --- DM: each scheme vs the best individual member, same window ----
        best_p, best_fz, best_arr = None, np.inf, None
        for p in members:
            Lm, Vm, Em = _to_loss(y, *members[p])
            rm, fzm = score(Lm, Vm, Em, p, cc, mask)
            if rm["FZ0"] < best_fz:
                best_fz, best_p, best_arr = rm["FZ0"], p, fzm
        print(f"  best individual member: {best_p} (FZ0={best_fz:+.4f})")
        for k, label in enumerate(trip):
            stat, p_one = V.dm_loss(fzs[(cc, label)], best_arr, nw_lag=H)
            rows[-3 + k]["DM_p_vs_best_member"] = round(p_one, 4)
            rows[-3 + k]["best_member"] = best_p
            # dm_loss: small p => the scheme is significantly BETTER than the
            # member; p near 1 => significantly WORSE. p in between => tie.
            verdict = ("beats member" if p_one < 0.05 else
                       "loses to member" if p_one > 0.95 else "tie")
            print(f"    DM {label:26s} p={p_one:.4f}  ({verdict})")

    out = pd.DataFrame(rows)
    os.makedirs(RESULT_DIR, exist_ok=True)
    out.to_excel(os.path.join(RESULT_DIR, "ens_fz0_audit.xlsx"), index=False)
    print("\n", out.to_string(index=False))
    print("\n-> result/ens_fz0_audit.xlsx")

    # LaTeX
    tex = [r"\begin{table}[!htbp]", r"\centering\footnotesize",
           r"\setlength{\tabcolsep}{5pt}",
           r"\caption{Combination schemes under the FZ0 criterion. Equal-weight "
           r"quantile pooling over-covers; FZ0-weighting restores calibration but "
           r"does not significantly beat the best individual member; the "
           r"in-sample-optimal combination is the worst out of sample.}",
           r"\label{tbl:ensembles}",
           r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Market & Scheme & FZ0 & Cov$_{95}$ & $p_{\mathrm{Kup}}$ & $R$\\ \midrule"]
    for cc in FILES:
        d = out[out.Market == cc]
        for i, (_, r) in enumerate(d.iterrows()):
            mk = cc if i == 0 else ""
            tex.append(f"{mk} & {r['Scheme']} & {r['FZ0']:+.3f} & {r['cov95']:.3f} "
                       f"& {r['Kupiec95']:.3f} & {r['ES_VaR']:.2f}\\\\")
        tex.append(r"\addlinespace[2pt]")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("_tbl_ensembles.tex", "w").write("\n".join(tex))
    print("-> _tbl_ensembles.tex")


if __name__ == "__main__":
    main()