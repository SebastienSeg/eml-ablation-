"""
targets.py — fonctions cibles pour l'évaluation.

Deux familles :

IN_DOMAIN   : fonctions dont l'expression exacte requiert exp ET log.
              EML est favorisé structurellement → test de l'hypothèse principale.

OUT_OF_DOMAIN : fonctions sans structure exp/log naturelle.
              EML devrait perdre ou égaliser → crédibilise les résultats in-domain
              en montrant que l'avantage n'est pas artificiel.
"""

import torch
from config import DELTA_DEFAULT

# ─────────────────────────────────────────────────────────────────────────────
# IN-DOMAIN  (requièrent exp ET log structurellement)
# ─────────────────────────────────────────────────────────────────────────────

def f_explog_product(x: torch.Tensor) -> torch.Tensor:
    """f(x,y) = exp(-x²) · ln(1+y²)
    Produit direct des deux primitives. Décomposable en un seul nœud EML.
    """
    return torch.exp(-x[:, 0]**2) * torch.log1p(x[:, 1]**2)


def f_power(x: torch.Tensor) -> torch.Tensor:
    """f(x,y) = exp(x · ln(y+1)) = (y+1)^x
    Composition stricte exp∘log. Irréductible sans les deux primitives.
    """
    return torch.exp(x[:, 0] * torch.log1p(torch.abs(x[:, 1])))


def f_logistic_log(x: torch.Tensor) -> torch.Tensor:
    """f(x,y,z) = sigmoid(x) · exp(y) · ln(z+2)
    Trois primitives imbriquées : sigmoid = exp-dérivé, exp explicite, log explicite.
    """
    return torch.sigmoid(x[:, 0]) * torch.exp(x[:, 1]) * torch.log(x[:, 2] + 2)


def f_cross_entropy(x: torch.Tensor) -> torch.Tensor:
    """f(x,y) = H(p(x)) + H(p(y))  avec p = sigmoid
    Entropie de Bernoulli — structure exp+log inévitable.
    """
    delta = DELTA_DEFAULT
    p = torch.sigmoid(x[:, 0])
    q = torch.sigmoid(x[:, 1])
    hp = -(p * torch.log(p + delta) + (1 - p) * torch.log(1 - p + delta))
    hq = -(q * torch.log(q + delta) + (1 - q) * torch.log(1 - q + delta))
    return hp + hq


# ─────────────────────────────────────────────────────────────────────────────
# OUT-OF-DOMAIN  (sans structure exp/log naturelle)
# ─────────────────────────────────────────────────────────────────────────────

def f_polynomial(x: torch.Tensor) -> torch.Tensor:
    """f(x,y) = x³ + 2x²y − 0.5y³ + 0.3xy
    Polynôme cubique — ReLU-MLP et KAN devraient dominer.
    """
    u, v = x[:, 0], x[:, 1]
    return u**3 + 2 * u**2 * v - 0.5 * v**3 + 0.3 * u * v


def f_trigonometric(x: torch.Tensor) -> torch.Tensor:
    """f(x,y) = sin(πx) · cos(πy) + 0.5 · sin(2πx)
    Série de Fourier tronquée — avantage aux primitives oscillantes (KAN/ReLU).
    """
    return (torch.sin(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1])
            + 0.5 * torch.sin(2 * torch.pi * x[:, 0]))


def f_piecewise(x: torch.Tensor) -> torch.Tensor:
    """f(x,y) = |x| · sign(y) + clamp(x·y, -1, 1)
    Fonction à non-linéarités abruptes — ReLU est exactement adapté.
    """
    return torch.abs(x[:, 0]) * torch.sign(x[:, 1]) + torch.clamp(x[:, 0] * x[:, 1], -1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES RÉELLES SIMULÉES  (décroissance radioactive bruitée)
# ─────────────────────────────────────────────────────────────────────────────

def generate_radioactive_decay(n: int = 300, noise_std: float = 0.05,
                               seed: int = 0) -> tuple:
    """N(t) = N₀ · exp(−λt) + ε,  ε ~ N(0, noise_std²)

    N₀ = 1.0, λ ~ Uniform(0.2, 2.0) selon le seed.
    Retourne (X, y) avec X = [t, λ] et y = N(t).

    Pourquoi ce choix :
    - Loi connue, analytiquement exp-pure → vérifie qu'EML gère bien l'exp
    - Le bruit gaussien ajoute une difficulté de généralisation réaliste
    - λ varie entre les samples pour éviter que le réseau mémorise une courbe unique
    """
    import numpy as np
    rng  = np.random.default_rng(seed)
    N0   = 1.0
    t    = torch.from_numpy(rng.uniform(0, 5, n)).float()
    lam  = torch.from_numpy(rng.uniform(0.2, 2.0, n)).float()
    noise = torch.from_numpy(rng.normal(0, noise_std, n)).float()
    y    = N0 * torch.exp(-lam * t) + noise
    X    = torch.stack([t, lam], dim=1)
    return X, y.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRE
# ─────────────────────────────────────────────────────────────────────────────

IN_DOMAIN: dict = {
    "explog_product": (f_explog_product, 2, (-2.0,  2.0)),
    "power_function":  (f_power,          2, ( 0.1,  2.0)),
    "logistic_log":    (f_logistic_log,   3, (-2.0,  2.0)),
    "cross_entropy":   (f_cross_entropy,  2, (-3.0,  3.0)),
}

OUT_OF_DOMAIN: dict = {
    "polynomial":    (f_polynomial,   2, (-2.0,  2.0)),
    "trigonometric": (f_trigonometric,2, (-1.0,  1.0)),
    "piecewise":     (f_piecewise,    2, (-2.0,  2.0)),
}
