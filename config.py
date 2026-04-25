"""
config.py — constantes globales et hyperparamètres par défaut.

"Topologie fixe" signifie que la structure du graphe est déterminée avant
l'entraînement et ne change pas pendant celui-ci. Seuls les poids scalaires
α, β et les matrices de projection W_exp / W_log sont appris par gradient.

Comparaison iso-paramétrique : HIDDEN définit la taille hidden de référence
pour ReLU-MLP. build_network() ajuste la taille hidden des autres variants
pour que le nombre total de paramètres soit aussi proche que possible de
ReLU-MLP. Cela garantit que les différences de performance sont attribuables
à la primitive, pas au budget paramétrique.
"""

import torch

# ── Reproductibilité ──────────────────────────────────────────────────────────
N_SEEDS_MAIN   = 25       # seeds pour les résultats principaux (publication)
N_SEEDS_SWEEP  = 5        # seeds pour le sweep HP (coût réduit)
GLOBAL_SEED    = 42

# ── Données ───────────────────────────────────────────────────────────────────
N_TRAIN        = 2000
N_TEST         = 500
N_NOISE_POINTS = 300      # points pour l'expérience "données réelles bruitées"

# ── Entraînement ──────────────────────────────────────────────────────────────
EPOCHS         = 3000
LR_DEFAULT     = 3e-3
CLIP_VAL       = 10.0
EPS_TARGET     = 1e-3     # seuil MSE pour mesure de steps-to-convergence

# ── Architecture ─────────────────────────────────────────────────────────────
HIDDEN         = 24       # taille hidden de référence (pour ReLU-MLP)
DEPTH          = 3        # nombre de couches cachées
# build_network() ajuste hidden par variant pour iso-paramétrie avec ReLU-MLP

# ── Numérique ─────────────────────────────────────────────────────────────────
DELTA_DEFAULT  = 1e-4     # terme de stabilité : ln(|x| + DELTA)
EXP_CLAMP     = 6.0       # clamp entrée de exp() pour éviter overflow

# ── Sweep hyperparamètres ─────────────────────────────────────────────────────
HP_SWEEP = {
    "lr":         [1e-3, 3e-3, 1e-2],
    "delta":      [1e-5, 1e-4, 1e-3],
    "init_scale": [0.1,  0.5,  1.0],   # écart-type init α et β
}

# ── Ablation δ (stabilité numérique du log) ──────────────────────────────────
DELTA_ABLATION = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
