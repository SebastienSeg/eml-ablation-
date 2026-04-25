"""
primitives.py — couches et architectures pour chaque variant.

Topologie fixe : toutes les architectures utilisent le même graphe
fully-connected (chaque neurone reçoit toutes les activations de la couche
précédente). Seule la fonction de transformation differ entre variants.
Les matrices de projection W_exp et W_log dans EMLLayer sont l'analogue
des matrices de poids classiques — elles définissent quelles combinaisons
linéaires des entrées alimentent respectivement la branche exp et la branche log.
"""

import torch
import torch.nn as nn
from config import DELTA_DEFAULT, EXP_CLAMP


# ─────────────────────────────────────────────────────────────────────────────
# EML
# ─────────────────────────────────────────────────────────────────────────────

class EMLLayer(nn.Module):
    """
    Nœud binaire : EML(x) = α · exp(W_exp · x) − β · ln(|W_log · x| + δ)

    Paramètres appris : W_exp (d_in × d_out), W_log (d_in × d_out),
                        α (d_out,), β (d_out,), bias (d_out,)

    freeze_exp=True → α fixé à 0 → variante "log-only interne" (EML-noexp)
    freeze_log=True → β fixé à 0 → variante "exp-only interne" (EML-nolog)

    init_scale contrôle l'écart-type de l'initialisation de α et β.
    """

    def __init__(self, in_features: int, out_features: int,
                 freeze_exp: bool = False, freeze_log: bool = False,
                 delta: float = DELTA_DEFAULT, init_scale: float = 1.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.delta        = delta
        self.freeze_exp   = freeze_exp
        self.freeze_log   = freeze_log

        self.W_exp = nn.Linear(in_features, out_features, bias=False)
        self.W_log = nn.Linear(in_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.empty(out_features).normal_(1.0, init_scale))
        self.beta  = nn.Parameter(torch.empty(out_features).normal_(1.0, init_scale))
        self.bias  = nn.Parameter(torch.zeros(out_features))

        nn.init.xavier_uniform_(self.W_exp.weight)
        nn.init.xavier_uniform_(self.W_log.weight)

        if freeze_exp:
            self.W_exp.weight.requires_grad_(False)
            self.alpha.data.zero_()
            self.alpha.requires_grad_(False)
        if freeze_log:
            self.W_log.weight.requires_grad_(False)
            self.beta.data.zero_()
            self.beta.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_part = torch.zeros(x.shape[0], self.out_features, device=x.device)
        log_part = torch.zeros_like(exp_part)
        if not self.freeze_exp:
            exp_part = self.alpha * torch.exp(
                torch.clamp(self.W_exp(x), -EXP_CLAMP, EXP_CLAMP)
            )
        if not self.freeze_log:
            log_part = self.beta * torch.log(
                torch.abs(self.W_log(x)) + self.delta
            )
        return exp_part - log_part + self.bias


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

class ExpOnlyLayer(nn.Module):
    """α · exp(W·x + b) — exponentielle pure."""
    def __init__(self, in_features: int, out_features: int, **_):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha  = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.exp(
            torch.clamp(self.linear(x), -EXP_CLAMP, EXP_CLAMP)
        )


class LogOnlyLayer(nn.Module):
    """α · ln(|W·x| + δ) — logarithme pur."""
    def __init__(self, in_features: int, out_features: int,
                 delta: float = DELTA_DEFAULT, **_):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha  = nn.Parameter(torch.ones(out_features))
        self.delta  = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.log(torch.abs(self.linear(x)) + self.delta)


class ReLULayer(nn.Module):
    """max(0, W·x + b) — MLP standard."""
    def __init__(self, in_features: int, out_features: int, **_):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


# ─────────────────────────────────────────────────────────────────────────────
# KAN-lite : activations B-spline apprenables
# ─────────────────────────────────────────────────────────────────────────────

class BSplineActivation(nn.Module):
    """Spline linéaire par morceaux à k points de contrôle apprenables."""

    def __init__(self, k: int = 8, x_min: float = -6.0, x_max: float = 6.0):
        super().__init__()
        self.register_buffer("grid", torch.linspace(x_min, x_max, k))
        self.ctrl = nn.Parameter(torch.randn(k) * 0.1)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.scale * (x - self.shift)
        grid = self.grid
        xc   = x_norm.clamp(grid[0], grid[-1] - 1e-6)
        idx  = torch.searchsorted(grid.contiguous(), xc.contiguous()).clamp(1, len(grid) - 1)
        t    = (xc - grid[idx - 1]) / (grid[idx] - grid[idx - 1] + 1e-8)
        return self.ctrl[idx - 1] * (1 - t) + self.ctrl[idx] * t


class KANLayer(nn.Module):
    """Couche KAN vectorisée : une B-spline apprise par connexion (i → j)."""

    def __init__(self, in_features: int, out_features: int,
                 k: int = 8, x_min: float = -6.0, x_max: float = 6.0, **_):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.k = k
        n_act = in_features * out_features

        self.register_buffer("grid", torch.linspace(x_min, x_max, k))
        self.ctrl  = nn.Parameter(torch.randn(n_act, k) * 0.1)
        self.scale = nn.Parameter(torch.ones(n_act))
        self.shift = nn.Parameter(torch.zeros(n_act))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # (B, in) → (B, out*in) : chaque entrée répétée pour chaque sortie
        x_exp  = x.repeat(1, self.out_features)
        x_norm = self.scale * (x_exp - self.shift)

        grid = self.grid
        xc  = x_norm.clamp(grid[0], grid[-1] - 1e-6)
        idx = torch.searchsorted(
            grid.contiguous(), xc.contiguous()
        ).clamp(1, self.k - 1)

        t = (xc - grid[idx - 1]) / (grid[idx] - grid[idx - 1] + 1e-8)

        # Gather des points de contrôle par activation
        ctrl_exp = self.ctrl.unsqueeze(0).expand(B, -1, -1)   # (B, n_act, k)
        ctrl_lo  = ctrl_exp.gather(2, (idx - 1).unsqueeze(2)).squeeze(2)
        ctrl_hi  = ctrl_exp.gather(2, idx.unsqueeze(2)).squeeze(2)

        y = ctrl_lo * (1 - t) + ctrl_hi * t                   # (B, n_act)
        return y.view(B, self.out_features, self.in_features).sum(dim=2)


# ─────────────────────────────────────────────────────────────────────────────
# Construction des réseaux
# ─────────────────────────────────────────────────────────────────────────────

def _eml_frozen_exp(i, o, **kw):
    return EMLLayer(i, o, freeze_exp=True,  **kw)

def _eml_frozen_log(i, o, **kw):
    return EMLLayer(i, o, freeze_log=True, **kw)


LAYER_MAP = {
    "EML":       EMLLayer,
    "EML-noexp": _eml_frozen_exp,
    "EML-nolog": _eml_frozen_log,
    "exp-only":  ExpOnlyLayer,
    "log-only":  LogOnlyLayer,
    "ReLU-MLP":  ReLULayer,
    "KAN":       KANLayer,
}

VARIANTS_MAIN    = list(LAYER_MAP.keys())
VARIANTS_DISPLAY = VARIANTS_MAIN  # ordre d'affichage


# ─────────────────────────────────────────────────────────────────────────────
# Iso-paramétrie : ajuster hidden par variant pour ~même n_params que ReLU-MLP
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_layer_params(variant: str, d_in: int, d_out: int) -> int:
    """Nombre total de paramètres pour une couche (y compris gelés)."""
    if variant in ("EML", "EML-noexp", "EML-nolog"):
        return 2 * d_in * d_out + 3 * d_out
    elif variant in ("exp-only", "log-only"):
        return d_in * d_out + 2 * d_out
    elif variant == "ReLU-MLP":
        return d_in * d_out + d_out
    elif variant == "KAN":
        return d_in * d_out * 10   # k=8 ctrl + scale + shift par activation
    return 0


def _estimate_network_params(variant: str, in_dim: int,
                             hidden: int, depth: int) -> int:
    """Nombre total de paramètres (y compris LayerNorm) pour un réseau."""
    dims = [in_dim] + [hidden] * depth + [1]
    total = 0
    for i in range(len(dims) - 1):
        total += _estimate_layer_params(variant, dims[i], dims[i + 1])
        if i < len(dims) - 2:
            total += 2 * dims[i + 1]
    return total


def _find_iso_hidden(variant: str, in_dim: int,
                     ref_hidden: int, depth: int) -> int:
    """Taille hidden qui rapproche le param count de celui de ReLU-MLP."""
    if variant == "ReLU-MLP":
        return ref_hidden
    ref_params = _estimate_network_params("ReLU-MLP", in_dim, ref_hidden, depth)
    best_h, best_diff = 1, float("inf")
    for h in range(1, ref_hidden * 3):
        diff = abs(_estimate_network_params(variant, in_dim, h, depth) - ref_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h
    return best_h


def build_network(variant: str, in_dim: int,
                  hidden: int, depth: int,
                  delta: float = DELTA_DEFAULT,
                  init_scale: float = 1.0) -> nn.Module:
    """
    Réseau profond avec la primitive spécifiée.

    Topologie : fully-connected, LayerNorm entre couches cachées.
    hidden est la taille de référence (pour ReLU-MLP). Les autres variants
    ajustent cette taille pour être iso-paramétriques (~même nombre total
    de paramètres que ReLU-MLP).
    """
    hidden = _find_iso_hidden(variant, in_dim, hidden, depth)
    LayerClass = LAYER_MAP[variant]
    layers = []
    dims   = [in_dim] + [hidden] * depth + [1]
    kw     = {"delta": delta, "init_scale": init_scale}

    for i in range(len(dims) - 1):
        layers.append(LayerClass(dims[i], dims[i + 1], **kw))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))

    return nn.Sequential(*layers)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
