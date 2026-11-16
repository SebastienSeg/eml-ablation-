"""
primitives.py — couches, registre de variants et construction des réseaux.

Architecture commune à tous les variants :

    x ─▶ [Primitive → LayerNorm] × depth ─▶ Linear(hidden → 1) ─▶ ŷ

Seule la primitive des couches cachées change d'un variant à l'autre. La
couche de sortie est TOUJOURS linéaire : une primitive en sortie biaiserait
la comparaison (ex. un ReLU en sortie ne peut pas produire de valeur < 0,
un exp ne peut pas changer de signe sans α).

Iso-paramétrie : la largeur `hidden` de chaque variant est choisie pour que
son nombre de paramètres ENTRAÎNABLES soit le plus proche possible de celui
de ReLU-MLP à largeur `cfg.hidden`. Le comptage se fait sur le vrai modèle
(pas de formule à la main qui pourrait diverger du code).

Ajouter un variant (ex. un "upgrade" d'EML) = écrire une fonction
`(d_in, d_out, layer_idx, cfg) -> nn.Module` décorée par `@register(...)`.
Il est alors automatiquement entraîné, iso-paramétré, testé statistiquement
contre EML et tracé dans les figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
import torch.nn as nn

from config import TrainConfig


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class Primitive(nn.Module):
    """Classe marqueur : les hooks de gradient ne s'attachent qu'à ces couches."""


class EMLLayer(Primitive):
    """
    EML(x) = α ⊙ exp(clamp(W_e·x + b_e)) − β ⊙ L(W_l·x + b_l) + bias

    log_mode :
      "abs"    → L(z) = ln(|z| + δ)        (version originale ; ∂L/∂z ≤ 1/δ)
      "smooth" → L(z) = ½·ln(z² + 1)       (lisse, ∂L/∂z ≤ ½, ≈ ln|z| pour |z| ≫ 1)
    branch_bias : ajoute b_e, b_l. Sans b_l, la singularité du log est
      collée à l'hyperplan W_l·x = 0, qui passe toujours par l'origine.
    use_exp / use_log : ablations. Une branche désactivée n'a aucun
      paramètre (équivalent à α ≡ 0 gelé, sans gaspiller de calcul).
    """

    def __init__(self, d_in: int, d_out: int, *, delta: float = 1e-4,
                 init_scale: float = 1.0, exp_clamp: float = 6.0,
                 use_exp: bool = True, use_log: bool = True,
                 branch_bias: bool = False, log_mode: str = "abs"):
        super().__init__()
        if not (use_exp or use_log):
            raise ValueError("au moins une branche doit être active")
        if log_mode not in ("abs", "smooth"):
            raise ValueError(f"log_mode inconnu : {log_mode!r}")
        self.delta, self.exp_clamp, self.log_mode = delta, exp_clamp, log_mode
        self.bias = nn.Parameter(torch.zeros(d_out))

        self.W_exp = self.alpha = None
        self.W_log = self.beta = None
        if use_exp:
            self.W_exp = nn.Linear(d_in, d_out, bias=branch_bias)
            self.alpha = nn.Parameter(torch.empty(d_out).normal_(1.0, init_scale))
            nn.init.xavier_uniform_(self.W_exp.weight)
        if use_log:
            self.W_log = nn.Linear(d_in, d_out, bias=branch_bias)
            self.beta = nn.Parameter(torch.empty(d_out).normal_(1.0, init_scale))
            nn.init.xavier_uniform_(self.W_log.weight)

    def log_branch(self, z: torch.Tensor) -> torch.Tensor:
        if self.log_mode == "abs":
            return torch.log(z.abs() + self.delta)
        return 0.5 * torch.log1p(z * z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bias
        if self.W_exp is not None:
            z = self.W_exp(x).clamp(-self.exp_clamp, self.exp_clamp)
            out = out + self.alpha * torch.exp(z)
        if self.W_log is not None:
            out = out - self.beta * self.log_branch(self.W_log(x))
        return out


class ExpOnlyLayer(Primitive):
    """α ⊙ exp(clamp(W·x + b))"""

    def __init__(self, d_in: int, d_out: int, *, exp_clamp: float = 6.0):
        super().__init__()
        self.linear, self.exp_clamp = nn.Linear(d_in, d_out), exp_clamp
        self.alpha = nn.Parameter(torch.ones(d_out))

    def forward(self, x):
        return self.alpha * torch.exp(self.linear(x).clamp(-self.exp_clamp, self.exp_clamp))


class LogOnlyLayer(Primitive):
    """α ⊙ ln(|W·x + b| + δ)"""

    def __init__(self, d_in: int, d_out: int, *, delta: float = 1e-4):
        super().__init__()
        self.linear, self.delta = nn.Linear(d_in, d_out), delta
        self.alpha = nn.Parameter(torch.ones(d_out))

    def forward(self, x):
        return self.alpha * torch.log(self.linear(x).abs() + self.delta)


class ReLULayer(Primitive):
    """max(0, W·x + b)"""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return torch.relu(self.linear(x))


class KANLayer(Primitive):
    """
    KAN-lite : y_j = Σ_i φ_ij(s_ij·(x_i − t_ij)), φ_ij spline linéaire par
    morceaux à k points de contrôle sur une grille uniforme fixe [x_min, x_max].
    Version simplifiée de Liu et al. 2024 (pas de B-splines cubiques, pas de
    terme de base SiLU, pas de raffinement de grille) → borne inférieure de KAN.
    """

    def __init__(self, d_in: int, d_out: int, *, k: int = 8,
                 x_min: float = -6.0, x_max: float = 6.0):
        super().__init__()
        self.d_in, self.d_out, self.k = d_in, d_out, k
        n_act = d_in * d_out
        self.register_buffer("grid", torch.linspace(x_min, x_max, k))
        self.ctrl  = nn.Parameter(torch.randn(n_act, k) * 0.1)
        self.scale = nn.Parameter(torch.ones(n_act))
        self.shift = nn.Parameter(torch.zeros(n_act))
        self.register_buffer("_row_offset", torch.arange(n_act) * k, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # activation a = j·d_in + i reçoit x_i
        xn = self.scale * (x.repeat(1, self.d_out) - self.shift)          # (B, n_act)
        g = self.grid
        h = g[1] - g[0]
        # grille uniforme → indice de l'intervalle calculé directement (pas de searchsorted)
        u = (xn.clamp(g[0], g[-1]) - g[0]) / h
        idx = u.detach().floor().clamp(0, self.k - 2).long()
        t = u - idx
        flat = self.ctrl.view(-1)
        base = idx + self._row_offset
        y = flat[base] * (1 - t) + flat[base + 1] * t
        return y.view(B, self.d_out, self.d_in).sum(dim=2)


# ─────────────────────────────────────────────────────────────────────────────
# Registre de variants
# ─────────────────────────────────────────────────────────────────────────────

LayerFactory = Callable[[int, int, int, TrainConfig], nn.Module]


@dataclass(frozen=True)
class Variant:
    name:        str
    family:      str          # "eml" | "ablation" | "baseline" | "upgrade"
    description: str
    make_layer:  LayerFactory # (d_in, d_out, layer_idx, cfg) → couche cachée


VARIANTS: dict[str, Variant] = {}


def register(name: str, family: str, description: str):
    def deco(fn: LayerFactory) -> LayerFactory:
        if name in VARIANTS:
            raise ValueError(f"variant déjà enregistré : {name}")
        VARIANTS[name] = Variant(name, family, description, fn)
        return fn
    return deco


def _eml_kw(cfg: TrainConfig) -> dict:
    return dict(delta=cfg.delta, init_scale=cfg.init_scale, exp_clamp=cfg.exp_clamp)


@register("EML", "eml", "α·exp(W_e·x) − β·ln(|W_l·x| + δ)")
def _eml(i, o, layer, cfg):
    return EMLLayer(i, o, **_eml_kw(cfg))


@register("EML-noexp", "ablation", "EML sans branche exp")
def _eml_noexp(i, o, layer, cfg):
    return EMLLayer(i, o, use_exp=False, **_eml_kw(cfg))


@register("EML-nolog", "ablation", "EML sans branche log")
def _eml_nolog(i, o, layer, cfg):
    return EMLLayer(i, o, use_log=False, **_eml_kw(cfg))


@register("exp-only", "baseline", "α·exp(W·x + b)")
def _exp_only(i, o, layer, cfg):
    return ExpOnlyLayer(i, o, exp_clamp=cfg.exp_clamp)


@register("log-only", "baseline", "α·ln(|W·x + b| + δ)")
def _log_only(i, o, layer, cfg):
    return LogOnlyLayer(i, o, delta=cfg.delta)


@register("ReLU-MLP", "baseline", "max(0, W·x + b)")
def _relu(i, o, layer, cfg):
    return ReLULayer(i, o)


@register("KAN", "baseline", "splines apprenables par connexion (k=8)")
def _kan(i, o, layer, cfg):
    return KANLayer(i, o)


# ── Upgrades ────────────────────────────────────────────────────────────────

@register("EML-v2", "upgrade",
          "EML + biais de branche + log lisse ½·ln(z²+1) (gradient borné)")
def _eml_v2(i, o, layer, cfg):
    return EMLLayer(i, o, branch_bias=True, log_mode="smooth", **_eml_kw(cfg))


@register("EML→ReLU", "upgrade",
          "hybride : 1re couche EML-v2, couches suivantes ReLU")
def _eml_hybrid(i, o, layer, cfg):
    if layer == 0:
        return EMLLayer(i, o, branch_bias=True, log_mode="smooth", **_eml_kw(cfg))
    return ReLULayer(i, o)


REFERENCE_VARIANT = "ReLU-MLP"


def variants_by_family(*families: str) -> list[str]:
    return [v.name for v in VARIANTS.values() if v.family in families]


# ─────────────────────────────────────────────────────────────────────────────
# Construction + iso-paramétrie
# ─────────────────────────────────────────────────────────────────────────────

def _assemble(variant: str, in_dim: int, hidden: int, cfg: TrainConfig) -> nn.Sequential:
    make = VARIANTS[variant].make_layer
    layers: list[nn.Module] = []
    d = in_dim
    for layer in range(cfg.depth):
        layers += [make(d, hidden, layer, cfg), nn.LayerNorm(hidden)]
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@lru_cache(maxsize=None)
def iso_hidden(variant: str, in_dim: int, ref_hidden: int, depth: int) -> int:
    """Largeur dont le nb de paramètres entraînables est le plus proche de ReLU-MLP."""
    cfg = TrainConfig(hidden=ref_hidden, depth=depth)
    if variant == REFERENCE_VARIANT:
        return ref_hidden
    target = count_params(_assemble(REFERENCE_VARIANT, in_dim, ref_hidden, cfg))
    candidates = range(1, 3 * ref_hidden + 1)
    return min(candidates,
               key=lambda h: abs(count_params(_assemble(variant, in_dim, h, cfg)) - target))


def build_network(variant: str, in_dim: int, cfg: TrainConfig) -> nn.Sequential:
    if variant not in VARIANTS:
        raise KeyError(f"variant inconnu : {variant!r} (connus : {list(VARIANTS)})")
    hidden = iso_hidden(variant, in_dim, cfg.hidden, cfg.depth)
    return _assemble(variant, in_dim, hidden, cfg)
