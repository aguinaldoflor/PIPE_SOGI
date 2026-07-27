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


def risk_objective(z, m, v, e, tau=TAU, lam=0.5):
    """Perda conjunta: pinball(mediana) ancora o ponto; FZ0 aprende a cauda.
    L = z - m é o resíduo realizado (diferenciável em m)."""
    L = z - m
    return lam * pinball(z, m, 0.5) + fz0_loss(L, v, e, tau)


# ------------------------------------------------------------------
# 2. Cabeça de risco — (m, v, e) com e >= v > 0 por construção
# ------------------------------------------------------------------
class RiskHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.f_m = nn.Linear(in_dim, 1)     # mediana (ponto)
        self.f_v = nn.Linear(in_dim, 1)     # -> gap do VaR (>0 via softplus)
        self.f_e = nn.Linear(in_dim, 1)     # -> gap do ES  (>0 via softplus)

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
        trend = F.avg_pool1d(x.unsqueeze(1), self.kernel, 1, padding=pad)
        trend = trend.squeeze(1)[:, :x.size(1)]
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
    return 0.5, True
