"""
targets.py — fonctions cibles et construction des datasets.

Familles (champ `family` de TargetSpec) :

in-domain      l'expression fait intervenir exp et/ou log, sur un petit domaine.
log-essential  le log est indispensable : entrées sur plusieurs ordres de
               grandeur (échantillonnage log-uniforme) ou structure multiplicative
               exp(Σ ln xᵢ). Un réseau sans log doit approcher ln par morceaux.
extrapolation  train et test sur des domaines DISJOINTS : seul un réseau qui a
               capturé la loi (et pas une approximation locale) prolonge juste.
ood            contrôle négatif : aucune structure exp/log.
extrapolation-control  contrôle négatif en extrapolation (polynôme).
radioactive    décroissance bruitée (loi exp + bruit), un dataset par seed.

Familles « favorables » (où l'hypothèse prédit un avantage EML) :
in-domain, log-essential, extrapolation, radioactive.
Familles « contrôles » (où elle n'en prédit aucun) : ood, extrapolation-control.

Tous les datasets sont standardisés (X et y, avec les statistiques du train).
La métrique principale est donc une NMSE = MSE / Var_train(y), comparable
entre cibles d'échelles très différentes et compatible avec un seuil de
convergence unique.
"""

from __future__ import annotations

import math
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
# LOG-ESSENTIAL (et extrapolation)
# ─────────────────────────────────────────────────────────────────────────────

def f_log_ratio(x: torch.Tensor) -> torch.Tensor:
    """ln(x₁ / x₂) = ln x₁ − ln x₂"""
    return torch.log(x[:, 0]) - torch.log(x[:, 1])


def f_product_ratio(x: torch.Tensor) -> torch.Tensor:
    """x₁·x₂ / x₃ = exp(ln x₁ + ln x₂ − ln x₃)"""
    return x[:, 0] * x[:, 1] / x[:, 2]


def f_power_law(x: torch.Tensor) -> torch.Tensor:
    """x₁^1.5 · x₂^−0.5 = exp(1.5 ln x₁ − 0.5 ln x₂)"""
    return x[:, 0] ** 1.5 / torch.sqrt(x[:, 1])


def f_decay(x: torch.Tensor) -> torch.Tensor:
    """exp(−λt), X = [t, λ]"""
    return torch.exp(-x[:, 1] * x[:, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Registre
# ─────────────────────────────────────────────────────────────────────────────

Box = tuple  # (lo, hi) pour toutes les dimensions, ou ((lo, hi), …) par dimension


@dataclass(frozen=True)
class TargetSpec:
    name:        str
    fn:          Callable[[torch.Tensor], torch.Tensor]
    in_dim:      int
    domain:      Box               # domaine d'entraînement
    data_seed:   int               # dataset identique pour tous les variants/seeds
    family:      str = "in-domain"
    log_uniform: bool = False      # échantillonnage uniforme en log (plusieurs décades)
    test_domain: Box | None = None # None = même domaine que le train

    def bounds(self, test: bool = False) -> list[tuple[float, float]]:
        box = self.test_domain if (test and self.test_domain is not None) else self.domain
        if isinstance(box[0], (int, float)):
            return [tuple(box)] * self.in_dim
        assert len(box) == self.in_dim, f"{self.name}: {len(box)} bornes pour {self.in_dim} dims"
        return [tuple(b) for b in box]


_SPECS = [
    TargetSpec("explog_product", f_explog_product, 2, (-2.0, 2.0), 101),
    TargetSpec("power_function", f_power,          2, ( 0.1, 2.0), 202),
    TargetSpec("logistic_log",   f_logistic_log,   3, (-2.0, 2.0), 303),
    TargetSpec("cross_entropy",  f_cross_entropy,  2, (-3.0, 3.0), 404),

    TargetSpec("polynomial",    f_polynomial,    2, (-2.0, 2.0), 505, "ood"),
    TargetSpec("trigonometric", f_trigonometric, 2, (-1.0, 1.0), 606, "ood"),
    TargetSpec("piecewise",     f_piecewise,     2, (-2.0, 2.0), 707, "ood"),

    TargetSpec("log_ratio",     f_log_ratio,     2, (1e-2, 1e2), 808, "log-essential", True),
    TargetSpec("product_ratio", f_product_ratio, 3, (0.5, 2.0),  909, "log-essential", True),
    TargetSpec("power_law",     f_power_law,     2, (0.2, 5.0),  1010, "log-essential", True),

    TargetSpec("extrap_log_ratio", f_log_ratio, 2, (0.1, 10.0), 1111, "extrapolation", True,
               test_domain=(10.0, 100.0)),
    TargetSpec("extrap_power", f_power, 2, (0.1, 1.0), 1212, "extrapolation",
               test_domain=(1.0, 2.0)),
    TargetSpec("extrap_product", f_product_ratio, 3, (0.5, 2.0), 1313, "extrapolation", True,
               test_domain=(2.0, 4.0)),
    TargetSpec("extrap_decay", f_decay, 2, ((0.0, 2.5), (0.2, 2.0)), 1414, "extrapolation",
               test_domain=((2.5, 5.0), (0.2, 2.0))),
    TargetSpec("extrap_polynomial", f_polynomial, 2, (-1.0, 1.0), 1515, "extrapolation-control",
               test_domain=(1.0, 2.0)),
]

TARGETS: dict[str, TargetSpec] = {s.name: s for s in _SPECS}
RADIOACTIVE = "radioactive_decay"

FAMILIES = ["in-domain", "log-essential", "extrapolation", "radioactive",
            "ood", "extrapolation-control"]
FAVORABLE_FAMILIES = ["in-domain", "log-essential", "extrapolation", "radioactive"]
CONTROL_FAMILIES = ["ood", "extrapolation-control"]


def targets_of(family: str) -> list[str]:
    if family == "radioactive":
        return [RADIOACTIVE]
    return [n for n, s in TARGETS.items() if s.family == family]


def family_of(target: str) -> str:
    return "radioactive" if target == RADIOACTIVE else TARGETS[target].family


IN_DOMAIN = {n: TARGETS[n] for n in targets_of("in-domain")}
OUT_OF_DOMAIN = {n: TARGETS[n] for n in targets_of("ood")}
ALL_TARGETS = [*TARGETS, RADIOACTIVE]


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


def _sample(spec: TargetSpec, n: int, test: bool, g: torch.Generator) -> torch.Tensor:
    cols = []
    for lo, hi in spec.bounds(test):
        u = torch.rand(n, generator=g)
        if spec.log_uniform:
            cols.append(torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
        else:
            cols.append(lo + u * (hi - lo))
    return torch.stack(cols, 1)


def make_dataset(spec: TargetSpec, n_train: int = N_TRAIN,
                 n_test: int = N_TEST) -> Dataset:
    """X et y standardisés avec les stats du TRAIN (y compris en extrapolation :
    le test peut alors sortir largement de [−3, 3], c'est voulu)."""
    g = torch.Generator().manual_seed(spec.data_seed)
    X_tr = _sample(spec, n_train, False, g)
    X_te = _sample(spec, n_test, True, g)
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
    spec = TARGETS.get(target)
    if spec is None:
        raise KeyError(f"cible inconnue : {target!r} (connues : {ALL_TARGETS})")
    return make_dataset(spec)


def target_in_dim(target: str) -> int:
    if target == RADIOACTIVE:
        return 2
    return TARGETS[target].in_dim
