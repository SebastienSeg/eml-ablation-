"""
training.py — boucle d'entraînement, collecte des métriques de publication.

Métriques collectées :
  - MSE train / test à convergence
  - Temps d'entraînement wall-clock
  - Divergence : run marqué divergent si loss devient NaN ou > 1e6
  - steps_to_eps : nombre d'epochs pour atteindre EPS_TARGET
  - Histogramme des gradients ∂loss/∂(sortie de couche) (optionnel, via hook)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from dataclasses import dataclass, field
from config import (DEVICE, EPOCHS, LR_DEFAULT, CLIP_VAL,
                    EPS_TARGET, HIDDEN, DEPTH)
from primitives import build_network, count_params


# ─────────────────────────────────────────────────────────────────────────────
# Résultat d'un run unique
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    variant:      str
    target:       str
    seed:         int
    final_mse_tr: float
    final_mse_te: float
    steps_to_eps: int          # −1 si jamais atteint dans EPOCHS
    n_params:     int
    time_sec:     float
    diverged:     bool         # True si NaN ou loss > 1e6
    loss_curve:   list = field(default_factory=list)   # échantillonné tous les 50 epochs
    grad_norms:   list = field(default_factory=list)   # norme L2 du gradient par epoch


# ─────────────────────────────────────────────────────────────────────────────
# Collecteur de gradients (hook sur les activations)
# ─────────────────────────────────────────────────────────────────────────────

class GradientCollector:
    """
    Enregistre les normes de ∂loss/∂(sortie de couche) via backward hook.
    Permet de construire des histogrammes pour vérifier absence de vanishing/exploding.
    """
    def __init__(self, model: nn.Module):
        self.records: list[float] = []
        self._hooks = []
        for name, layer in model.named_modules():
            if hasattr(layer, "weight") or hasattr(layer, "alpha"):
                h = layer.register_full_backward_hook(self._hook)
                self._hooks.append(h)

    def _hook(self, module, grad_in, grad_out):
        # grad_out = dL/d(sortie du module), pas dL/dx.
        # Pour l'analyse vanishing/exploding, RunResult.grad_norms
        # (norme globale des gradients des parametres) est complementaire.
        for g in grad_out:
            if g is not None:
                self.records.append(float(g.norm().item()))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def summary(self) -> dict:
        if not self.records:
            return {}
        arr = np.array(self.records)
        return {
            "mean":  float(arr.mean()),
            "std":   float(arr.std()),
            "p5":    float(np.percentile(arr, 5)),
            "p95":   float(np.percentile(arr, 95)),
            "pct_vanishing":  float((arr < 1e-6).mean()),  # quasi-nul
            "pct_exploding":  float((arr > 1e2).mean()),   # trop grand
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run unique
# ─────────────────────────────────────────────────────────────────────────────

def run_one(variant: str, target: str, seed: int,
            X_tr: torch.Tensor, y_tr: torch.Tensor,
            X_te: torch.Tensor, y_te: torch.Tensor,
            lr: float = LR_DEFAULT,
            delta: float = 1e-4,
            init_scale: float = 1.0,
            collect_grads: bool = False) -> RunResult:
    """
    Entraîne un réseau et retourne toutes les métriques de publication.

    X_tr, y_tr, X_te, y_te sont précalculés à l'extérieur pour garantir
    que tous les variants voient exactement les mêmes données.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim  = X_tr.shape[1]
    model   = build_network(variant, in_dim, HIDDEN, DEPTH,
                            delta=delta, init_scale=init_scale).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.MSELoss()

    n_params     = count_params(model)
    loss_curve   = []
    grad_norms   = []
    steps_to_eps = -1
    diverged     = False
    t0           = time.time()

    # Hook de gradients (optionnel, coûteux — activé seulement pour EML)
    grad_col = GradientCollector(model) if collect_grads else None

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()

        try:
            pred = model(X_tr)
            loss = loss_fn(pred, y_tr)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                diverged = True
                break

            loss.backward()
            total_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CLIP_VAL)
            opt.step()
        except RuntimeError:
            diverged = True
            break

        sched.step()
        grad_norms.append(float(total_grad_norm))

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                mse_te = loss_fn(model(X_te), y_te).item()
            loss_curve.append(float(loss.item()))
            if mse_te < EPS_TARGET and steps_to_eps == -1:
                steps_to_eps = epoch
            model.train()

    # Métriques finales
    model.eval()
    if not diverged:
        with torch.no_grad():
            try:
                final_mse_tr = loss_fn(model(X_tr), y_tr).item()
                final_mse_te = loss_fn(model(X_te), y_te).item()
            except RuntimeError:
                final_mse_tr = final_mse_te = float("inf")
    else:
        final_mse_tr = final_mse_te = float("inf")

    if grad_col:
        grad_col.remove()

    return RunResult(
        variant=variant, target=target, seed=seed,
        final_mse_tr=final_mse_tr, final_mse_te=final_mse_te,
        steps_to_eps=steps_to_eps, n_params=n_params,
        time_sec=time.time() - t0, diverged=diverged,
        loss_curve=loss_curve, grad_norms=grad_norms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Collecte des gradients EML (un seul seed, pour l'histogramme)
# ─────────────────────────────────────────────────────────────────────────────

def collect_gradient_histogram(variant: str, target: str,
                                X_tr, y_tr, X_te, y_te,
                                seed: int = 0) -> dict:
    """
    Retourne le résumé statistique des gradients ∂loss/∂(sortie de couche)
    sur un entraînement complet. Utilisé pour la figure d'histogramme.
    """
    torch.manual_seed(seed)
    in_dim  = X_tr.shape[1]
    model   = build_network(variant, in_dim, HIDDEN, DEPTH).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR_DEFAULT)
    loss_fn = nn.MSELoss()
    col     = GradientCollector(model)

    for epoch in range(min(EPOCHS, 500)):  # 500 epochs suffisent pour l'histo
        model.train()
        opt.zero_grad()
        try:
            loss = loss_fn(model(X_tr), y_tr)
            if torch.isnan(loss):
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_VAL)
            opt.step()
        except RuntimeError:
            break

    col.remove()
    return col.summary()
