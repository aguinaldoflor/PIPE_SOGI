"""
FZ0-Deep-Risk — esqueleto da Contribuição 1 da tese
====================================================
Rede que aprende o par (VaR, ES) de forma end-to-end sob a perda de
Fissler-Ziegel (0-homogênea), colapsando o pipeline de duas etapas
(ponto + VaR econométrico) numa única arquitetura treinada com objetivo
estritamente consistente para o risco.

Convenção de procurement (cauda superior da perda de compra):
  z_t = log-preço escalado realizado em t+h
  m_t = mediana prevista (ponto)
  L_t = z_t - m_t          (resíduo = perda realizada vs. mediana)
  v_t = VaR_tau(L)  > 0     (buffer)
  e_t = ES_tau(L)   >= v_t  (severidade média além do VaR)

Posicionamento na literatura (para o paper):
  - Fissler & Ziegel (2016): elicitabilidade conjunta de (VaR, ES).
  - Patton, Ziegel & Chen (2019) e Taylor (2019): estimação sob perda FZ,
    mas restrita a modelos semiparamétricos/lineares de baixa dimensão.
  - LACUNA (a contribuição): usar o objetivo FZ0 para treinar arquiteturas
    de deep forecasting end-to-end, com recalibração conformal, comparadas
    entre mercados de commodities.
  - Zeng et al. (2023, DLinear): backbone linear simples como âncora do debate
    "Transformers são eficazes para séries temporais?".
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TAU = 0.95          # nível de VaR/ES (cauda superior)


# ------------------------------------------------------------------
# 1. Perda FZ0 diferenciável (objetivo de treino)
# ------------------------------------------------------------------
def fz0_loss(L, v, e, tau=TAU, eps=1e-4):
    """0-homogeneous Fissler-Ziegel loss para o par (VaR, ES), cauda superior.
    L,(v,e): tensores (B,). e é pisado em eps para estabilizar ln(e) e 1/e.
    O indicador 1{L>=v} tem subgradiente bem-definido (como o kink do pinball)."""
    e = e.clamp_min(eps)
    ind = (L >= v).float()
    term1 = ind * (L - v) / ((1.0 - tau) * e)
    term2 = v / e
    term3 = torch.log(e) - 1.0
    return (term1 + term2 + term3).mean()


def pinball(z, q, tau):
    d = z - q
    return torch.maximum(tau * d, (tau - 1.0) * d).mean()


def risk_objective(z, m, v, e, tau=TAU, lam=1.0):
    """Perda conjunta com a mediana DESACOPLADA do termo FZ0.

    A perda FZ0, com a mediana livre, tem um mínimo degenerado: enviesar m para
    cima elimina as exceções (cobertura -> 1) e o ES colapsa em zero explorando
    ln(e) -> -inf. Para evitar isso, m é treinada exclusivamente pela pinball da
    mediana, e (v, e) pela FZ0 sobre o RESÍDUO com m destacado (`m.detach()`).
    Assim o resíduo fica centrado (~1-tau de exceções reais), o que reativa o
    termo de exceção que penaliza e -> 0. A rede continua emitindo (m, v, e)
    conjuntamente; apenas o acoplamento degenerado m<-FZ0 é removido."""
    L = z - m.detach()
    return lam * pinball(z, m, 0.5) + fz0_loss(L, v, e, tau)


# ------------------------------------------------------------------
# 2. Cabeça de risco — (m, v, e) com e >= v > 0 por construção
# ------------------------------------------------------------------
class RiskHead(nn.Module):
    def __init__(self, in_dim, init_gap=-2.5):
        super().__init__()
        self.f_m = nn.Linear(in_dim, 1)     # mediana (ponto)
        self.f_v = nn.Linear(in_dim, 1)     # -> gap do VaR (>0 via softplus)
        self.f_e = nn.Linear(in_dim, 1)     # -> gap do ES  (>0 via softplus)
        # Inicia as bandas ESTREITAS: softplus(0)=0.69 é enorme vs. a escala do
        # resíduo (~0.02) e produziria cobertura ~1 sem exceções para treinar.
        # softplus(-2.5) ~ 0.08 parte estreito; o warmup ajusta ao quantil real.
        nn.init.constant_(self.f_v.bias, init_gap)
        nn.init.constant_(self.f_e.bias, init_gap)

    def forward(self, h):
        m = self.f_m(h).squeeze(-1)
        v = F.softplus(self.f_v(h)).squeeze(-1)         # VaR do resíduo > 0
        e = v + F.softplus(self.f_e(h)).squeeze(-1)     # ES >= VaR (monotônico)
        return m, v, e


# ------------------------------------------------------------------
# 3. Backbone DLinear (mínimo, vencedor nos seus dados) + cabeça de risco
# ------------------------------------------------------------------
class FZ0DLinear(nn.Module):
    """DLinear: decomposição tendência/sazonal por média móvel + mapas lineares.
    Trocar este backbone por N-HiTS/TSMixer é a ablação de arquitetura."""
    def __init__(self, input_len, n_cov=0, kernel=25):
        super().__init__()
        self.input_len = input_len
        self.kernel = kernel if kernel % 2 == 1 else kernel + 1
        self.lin_trend = nn.Linear(input_len, input_len)
        self.lin_season = nn.Linear(input_len, input_len)
        self.head = RiskHead(input_len + n_cov)

    def _decomp(self, x):                                # x: (B, input_len)
        pad = self.kernel // 2
        # replicate-pad the ends (as in DLinear), then moving average with no
        # extra padding -> unbiased trend at the boundaries (avg_pool1d's
        # default zero-padding would pull the edge trend toward zero).
        xp = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        trend = F.avg_pool1d(xp, self.kernel, 1, padding=0).squeeze(1)[:, :x.size(1)]
        return trend, x - trend                          # tendência, sazonal

    def forward(self, x, cov=None):                      # cov: (B, n_cov) no origin t
        tr, se = self._decomp(x)
        feat = self.lin_trend(tr) + self.lin_season(se)  # (B, input_len)
        if cov is not None:
            feat = torch.cat([feat, cov], dim=1)
        return self.head(feat)                           # m, v, e


# ------------------------------------------------------------------
# 4. Passo de treino (com clip — FZ0 pode dar gradiente instável)
# ------------------------------------------------------------------
def train_step(model, x, cov, z, opt, tau=TAU, lam=0.5, clip=1.0):
    model.train()
    m, v, e = model(x, cov)
    loss = risk_objective(z, m, v, e, tau, lam)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # estabilidade FZ0
    opt.step()
    return float(loss.detach())


# ------------------------------------------------------------------
# 5. Curriculum opcional: aquece com pinball, depois liga o FZ0
#    (mitiga instabilidade inicial de ln(e)/1/e quando e ~ 0)
# ------------------------------------------------------------------
def warmup_then_fz0(epoch, warmup=20):
    """Retorna (lam_ponto, usar_fz0). Nas primeiras `warmup` épocas treina só a
    mediana + pinball dos gaps para estabilizar antes de ligar o FZ0."""
    if epoch < warmup:
        return 1.0, False
    return 1.0, True         # âncora forte na mediana durante a fase FZ0


# ------------------------------------------------------------------
# 6. Cola com os dados — reaproveita o preprocess do pipeline_v2p
# ------------------------------------------------------------------
def make_windows(d, input_len, h):
    """Constrói amostras (x, cov, z) do protocolo direto de h passos a partir do
    dict retornado por pipeline_v2p.preprocess(...). Split por índice do ALVO
    (t+h): train se alvo < t1; val se t1<=alvo<t2; test se alvo>=t2 — sem
    vazamento. x = janela de `input_len` do alvo escalado; cov = covariáveis na
    origem t; z = alvo escalado em t+h.
    Retorna dict com arrays numpy (X, C, Z) por partição + o índice de teste."""
    z_all = d["ssa"].univariate_values().astype("float32")     # alvo escalado (T,)
    c_all = d["covs"].values().astype("float32")                # covariáveis (T, n_cov)
    tidx = d["ssa"].time_index
    T = len(z_all); t1, t2 = d["t1"], d["t2"]

    X, C, Z, TT = [], [], [], []
    for tt in range(input_len + h, T):          # tt = índice do alvo
        o = tt - h                               # origem
        if o - input_len + 1 < 0:
            continue
        X.append(z_all[o - input_len + 1: o + 1])
        C.append(c_all[o])
        Z.append(z_all[tt]); TT.append(tt)
    X = np.asarray(X); C = np.asarray(C); Z = np.asarray(Z); TT = np.asarray(TT)

    tr = TT < t1; va = (TT >= t1) & (TT < t2); te = TT >= t2
    return dict(
        Xtr=X[tr], Ctr=C[tr], Ztr=Z[tr],
        Xva=X[va], Cva=C[va], Zva=Z[va],
        Xte=X[te], Cte=C[te], Zte=Z[te],
        test_time=tidx[TT[te]], n_cov=C.shape[1], input_len=input_len)


def _fz0_np(L, v, e, tau=TAU, eps=1e-4):
    e = np.maximum(e, eps); ind = (L >= v).astype(float)
    return float(np.mean(ind * (L - v) / ((1 - tau) * e) + v / e + np.log(e) - 1))


def fit(model, w, epochs=300, lr=1e-3, warmup=20, patience=25,
        batch=256, tau=TAU, device="cpu", verbose=True):
    """Treina com curriculum (pinball -> FZ0) e early stopping pela FZ0 de
    validação. Retorna o modelo com os melhores pesos (menor FZ0 de val)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=10)
    Xtr = torch.tensor(w["Xtr"], device=device); Ctr = torch.tensor(w["Ctr"], device=device)
    Ztr = torch.tensor(w["Ztr"], device=device)
    Xva = torch.tensor(w["Xva"], device=device); Cva = torch.tensor(w["Cva"], device=device)
    n = len(Xtr); best = (1e18, None); bad = 0
    for ep in range(epochs):
        lam, use_fz0 = warmup_then_fz0(ep, warmup)
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            m, v, e = model(Xtr[j], Ctr[j])
            if use_fz0:
                loss = risk_objective(Ztr[j], m, v, e, tau, lam)
            else:                                # warmup: median + VaR/ES of the RESIDUAL
                resid = (Ztr[j] - m).detach()    # detach so (v,e) do not fight m
                # full-weight pinball so v is placed at the residual's tau-quantile
                # (and e slightly beyond) BEFORE the FZ0 phase refines them.
                loss = (pinball(Ztr[j], m, 0.5)
                        + pinball(resid, v, tau)
                        + pinball(resid, e, 0.5 * (1 + tau)))   # e ~ higher quantile
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # validação pelo OBJETIVO COMPOSTO (pinball da mediana + FZ0 do resíduo).
        # Selecionar época por FZ0 sozinho é degenerado: uma mediana que deriva
        # para cima zera os resíduos positivos, encolhe (v,e) e "melhora" o FZ0
        # espuriamente — a seleção travaria nas épocas enviesadas.
        model.eval()
        with torch.no_grad():
            mv, vv, ev = model(Xva, Cva)
        mv_np = mv.cpu().numpy(); resid = w["Zva"] - mv_np
        val_fz = _fz0_np(resid, vv.cpu().numpy(), ev.cpu().numpy(), tau)
        val_pin = 0.5 * float(np.mean(np.abs(resid)))        # pinball(0.5) da mediana
        med_bal = float(np.mean(resid > 0))                   # ~0.5 se mediana sadia
        val = val_pin * 10.0 + val_fz                         # composto (pinball em escala)
        sched.step(val)
        if val < best[0] - 1e-6:
            best = (val, {k: t.detach().clone() for k, t in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if verbose and ep % 20 == 0:
            print(f"  ep {ep:3d} | val={val:.4f} (fz={val_fz:.3f} pin={val_pin:.4f} "
                  f"med_bal={med_bal:.2f}) | best={best[0]:.4f}")
        if bad >= patience and ep > warmup:
            break
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best[0]


def predict(model, w, device="cpu"):
    """Prevê (m, v, e) no conjunto de TESTE. m,v,e no espaço escalado do alvo;
    v,e são VaR/ES do resíduo L=z-m. Retorna arrays numpy."""
    model = model.to(device).eval()
    with torch.no_grad():
        m, v, e = model(torch.tensor(w["Xte"], device=device),
                        torch.tensor(w["Cte"], device=device))
    return m.cpu().numpy(), v.cpu().numpy(), e.cpu().numpy()
