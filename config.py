"""
config.py — hyperparamètres par défaut de l'étude.

Tout ce qui pilote un entraînement est regroupé dans `TrainConfig` (dataclass
immuable) : un run est entièrement décrit par (variant, cible, seed, TrainConfig),
ce qui rend chaque résultat sérialisable, reproductible et reprenable.

Les constantes module (N_SEEDS_MAIN, HP_SWEEP…) décrivent le *protocole*
expérimental, pas un run individuel.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import torch

# ── Protocole ────────────────────────────────────────────────────────────────
N_SEEDS_MAIN   = 25       # seeds pour les résultats principaux
N_SEEDS_QUICK  = 5        # seeds en mode --quick
N_SEEDS_SWEEP  = 5        # seeds pour le sweep HP et l'ablation δ

N_TRAIN        = 2000
N_TEST         = 500

# Sweep hyperparamètres (variant EML uniquement)
HP_SWEEP = {
    "lr":         [1e-3, 3e-3, 1e-2],
    "delta":      [1e-5, 1e-4, 1e-3],
    "init_scale": [0.1,  0.5,  1.0],
}

# Ablation δ (stabilité numérique du log)
DELTA_ABLATION = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# Seuils
DIVERGENCE_LOSS = 1e6     # loss (sur y standardisé) au-delà de laquelle le run est divergent
EPS_TARGET      = 1e-3    # NMSE train à atteindre pour "steps_to_eps"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparamètres d'un run. Hashable → sert de clé de cache/reprise."""
    epochs:     int   = 3000
    lr:         float = 3e-3
    clip:       float = 10.0      # norme max du gradient (clip_grad_norm_)
    hidden:     int   = 24        # taille hidden de référence (ReLU-MLP)
    depth:      int   = 3         # nombre de couches cachées
    delta:      float = 1e-4      # ln(|z| + δ) dans la branche log
    init_scale: float = 1.0       # écart-type de l'init de α, β : N(1, init_scale²)
    exp_clamp:  float = 6.0       # exp(clamp(z, ±exp_clamp))
    log_every:  int   = 50        # échantillonnage de la courbe de loss

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "TrainConfig":
        return replace(self, **kw)
