"""
targets.py — fonctions cibles et construction des datasets.

Trois familles :

IN_DOMAIN     : l'expression exacte fait intervenir exp ET/OU log.
                Hypothèse : EML devrait y être avantagé.
OUT_OF_DOMAIN : aucune structure exp/log. Contrôle négatif : si EML y gagne
                aussi, l'avantage in-domain n'est pas imputable à la primitive.
NOISY         : décroissance radioactive bruitée (loi exp pure + bruit).

Tous les datasets sont standardisés (X et y, avec les statistiques du train).
La métrique principale est donc une NMSE = MSE / Var_train(y), comparable
entre cibles d'échelles très différentes et compatible avec un seuil de
convergence unique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from config import N_TRAIN, N_TEST


# ─────────────────────────────────────────────────────────────────────────────
# IN-DOMAIN
# ─────────────────────────────────────────────────────────────────────────────

def f_explog_product(x: torch.Tensor) -> torch.Tensor:
    """exp(−x²) · ln(1 + y²)"""
    return torch.exp(-x[:, 0] ** 2) * torch.log1p(x[:, 1] ** 2)


def f_power(x: torch.Tensor) -> torch.Tensor:
    """(y + 1)^x = exp(x · ln(y + 1)),  y > 0 sur le domaine utilisé."""
    return torch.exp(x[:, 0] * torch.log1p(x[:, 1]))


def f_logistic_log(x: torch.Tensor) -> torch.Tensor:
    """sigmoid(x) · exp(y) · ln(z + 2)"""
    return torch.sigmoid(x[:, 0]) * torch.exp(x[:, 1]) * torch.log(x[:, 2] + 2)


def _bernoulli_entropy_from_logit(z: torch.Tensor) -> torch.Tensor:
    """H(σ(z)) exacte et stable : softplus(z) − z·σ(z)."""
    return F.softplus(z) - z * torch.sigmoid(z)


def f_cross_entropy(x: torch.Tensor) -> torch.Tensor:
    """H(σ(x)) + H(σ(y)), H = entropie de Bernoulli."""
    return _bernoulli_entropy_from_logit(x[:, 0]) + _bernoulli_entropy_from_logit(x[:, 1])


# ─────────────────────────────────────────────────────────────────────────────
# OUT-OF-DOMAIN
# ─────────────────────────────────────────────────────────────────────────────

def f_polynomial(x: torch.Tensor) -> torch.Tensor:
    """x³ + 2x²y − 0.5y³ + 0.3xy"""
    u, v = x[:, 0], x[:, 1]
    return u ** 3 + 2 * u ** 2 * v - 0.5 * v ** 3 + 0.3 * u * v


def f_trigonometric(x: torch.Tensor) -> torch.Tensor:
    """sin(πx)·cos(πy) + 0.5·sin(2πx)"""
    return (torch.sin(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1])
            + 0.5 * torch.sin(2 * torch.pi * x[:, 0]))


def f_piecewise(x: torch.Tensor) -> torch.Tensor:
    """|x|·sign(y) + clamp(x·y, −1, 1)"""
    return torch.abs(x[:, 0]) * torch.sign(x[:, 1]) + torch.clamp(x[:, 0] * x[:, 1], -1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Registre
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TargetSpec:
    name:      str
    fn:        Callable[[torch.Tensor], torch.Tensor]
    in_dim:    int
    domain:    tuple[float, float]
    data_seed: int                 # dataset identique pour tous les variants/seeds


IN_DOMAIN: dict[str, TargetSpec] = {s.name: s for s in [
    TargetSpec("explog_product", f_explog_product, 2, (-2.0, 2.0), 101),
    TargetSpec("power_function", f_power,          2, ( 0.1, 2.0), 202),
    TargetSpec("logistic_log",   f_logistic_log,   3, (-2.0, 2.0), 303),
    TargetSpec("cross_entropy",  f_cross_entropy,  2, (-3.0, 3.0), 404),
]}

OUT_OF_DOMAIN: dict[str, TargetSpec] = {s.name: s for s in [
    TargetSpec("polynomial",    f_polynomial,    2, (-2.0, 2.0), 505),
    TargetSpec("trigonometric", f_trigonometric, 2, (-1.0, 1.0), 606),
    TargetSpec("piecewise",     f_piecewise,     2, (-2.0, 2.0), 707),
]}

RADIOACTIVE = "radioactive_decay"
ALL_TARGETS = [*IN_DOMAIN, *OUT_OF_DOMAIN, RADIOACTIVE]


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Dataset:
    """Données standardisées + statistiques pour revenir à l'échelle d'origine."""
    X_tr: torch.Tensor
    y_tr: torch.Tensor
    X_te: torch.Tensor
    y_te: torch.Tensor
    y_mean: float
    y_std:  float

    def to(self, device: str) -> "Dataset":
        return Dataset(self.X_tr.to(device), self.y_tr.to(device),
                       self.X_te.to(device), self.y_te.to(device),
                       self.y_mean, self.y_std)


def _standardize(X_tr, y_tr, X_te, y_te) -> Dataset:
    x_mu, x_sd = X_tr.mean(0), X_tr.std(0).clamp_min(1e-8)
    y_mu, y_sd = y_tr.mean(), y_tr.std().clamp_min(1e-8)
    return Dataset(
        (X_tr - x_mu) / x_sd, ((y_tr - y_mu) / y_sd).unsqueeze(1),
        (X_te - x_mu) / x_sd, ((y_te - y_mu) / y_sd).unsqueeze(1),
        float(y_mu), float(y_sd),
    )


def make_dataset(spec: TargetSpec, n_train: int = N_TRAIN,
                 n_test: int = N_TEST) -> Dataset:
    g = torch.Generator().manual_seed(spec.data_seed)
    lo, hi = spec.domain
    X_tr = torch.empty(n_train, spec.in_dim).uniform_(lo, hi, generator=g)
    X_te = torch.empty(n_test,  spec.in_dim).uniform_(lo, hi, generator=g)
    return _standardize(X_tr, spec.fn(X_tr), X_te, spec.fn(X_te))


def make_radioactive_dataset(seed: int, n_train: int = N_TRAIN,
                             n_test: int = N_TEST,
                             noise_std: float = 0.05) -> Dataset:
    """N(t) = exp(−λt) + ε,  X = [t, λ],  t ~ U(0, 5),  λ ~ U(0.2, 2).

    Le bruit n'est ajouté qu'au train : le test mesure la capacité à
    retrouver la loi sous-jacente, pas à prédire le bruit (sinon la MSE
    test aurait un plancher irréductible de noise_std²).
    Un dataset différent par seed : la variance mesurée inclut celle des données.
    """
    g = torch.Generator().manual_seed(10_000 + seed)

    def sample(n):
        t   = torch.empty(n).uniform_(0, 5, generator=g)
        lam = torch.empty(n).uniform_(0.2, 2.0, generator=g)
        return torch.stack([t, lam], 1), torch.exp(-lam * t)

    X_tr, y_tr = sample(n_train)
    y_tr = y_tr + noise_std * torch.randn(n_train, generator=g)
    X_te, y_te = sample(n_test)
    return _standardize(X_tr, y_tr, X_te, y_te)


def get_dataset(target: str, seed: int) -> Dataset:
    if target == RADIOACTIVE:
        return make_radioactive_dataset(seed)
    spec = IN_DOMAIN.get(target) or OUT_OF_DOMAIN.get(target)
    if spec is None:
        raise KeyError(f"cible inconnue : {target!r} (connues : {ALL_TARGETS})")
    return make_dataset(spec)


def target_in_dim(target: str) -> int:
    if target == RADIOACTIVE:
        return 2
    return (IN_DOMAIN.get(target) or OUT_OF_DOMAIN[target]).in_dim
