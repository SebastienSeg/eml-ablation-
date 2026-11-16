"""
training.py — entraînement d'un run et métriques.

Un run = (variant, cible, seed, TrainConfig). Entraînement full-batch Adam +
cosine annealing + clip_grad_norm_, sur données standardisées.

Métriques (toutes sur y standardisé, donc des NMSE) :
  nmse_tr / nmse_te  : MSE finale train / test
  best_nmse_te       : meilleure NMSE test vue aux points d'échantillonnage
                       (diagnostic seulement, ne pas l'utiliser pour comparer :
                       c'est une sélection sur le test)
  steps_to_eps       : 1re epoch où la loss train < EPS_TARGET (−1 sinon)
  diverged           : loss NaN/inf ou > DIVERGENCE_LOSS → nmse = inf
  clip_rate          : fraction des steps où le gradient a été clippé
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn

from config import DEVICE, DIVERGENCE_LOSS, EPS_TARGET, TrainConfig
from primitives import Primitive, build_network, count_params
from targets import Dataset, get_dataset


@dataclass
class RunResult:
    variant:      str
    target:       str
    seed:         int
    tag:          str            # expérience : "main", "hp_sweep", "delta", …
    config:       dict           # TrainConfig.to_dict()
    nmse_tr:      float
    nmse_te:      float
    best_nmse_te: float
    steps_to_eps: int
    n_params:     int
    time_sec:     float
    diverged:     bool
    clip_rate:    float
    loss_curve:   list = field(default_factory=list)   # loss train tous les log_every
    grad_samples: dict = field(default_factory=dict)   # {couche: [‖∂L/∂out‖, …]}

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("nmse_tr", "nmse_te", "best_nmse_te"):   # JSON n'a pas inf
            if not math.isfinite(d[k]):
                d[k] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        d = dict(d)
        for k in ("nmse_tr", "nmse_te", "best_nmse_te"):
            if d[k] is None:
                d[k] = float("inf")
        return cls(**d)

    @property
    def key(self) -> tuple:
        return run_key(self.tag, self.variant, self.target, self.seed,
                       TrainConfig(**self.config))


def run_key(tag: str, variant: str, target: str, seed: int, cfg: TrainConfig) -> tuple:
    return (tag, variant, target, seed, tuple(sorted(cfg.to_dict().items())))


class GradientCollector:
    """‖∂L/∂(sortie)‖ pour chaque couche Primitive, tous les `every` steps."""

    def __init__(self, model: nn.Module, every: int = 10):
        self.every, self.step = every, 0
        self.samples: dict[str, list[float]] = {}
        self._hooks = [
            m.register_forward_hook(self._make_hook(name))
            for name, m in model.named_modules() if isinstance(m, Primitive)
        ]

    def _make_hook(self, name: str):
        def record(grad: torch.Tensor):
            self.samples.setdefault(name, []).append(float(grad.norm()))

        def hook(_module, _inp, out):
            if out.requires_grad and self.step % self.every == 0:
                out.register_hook(record)
        return hook

    def tick(self):
        self.step += 1

    def remove(self):
        for h in self._hooks:
            h.remove()


def _mse(model, X, y) -> float:
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(X), y))


def run_one(variant: str, target: str, seed: int, cfg: TrainConfig,
            tag: str = "main", data: Dataset | None = None,
            collect_grads: bool = False) -> RunResult:
    if data is None:
        data = get_dataset(target, seed)
    data = data.to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_network(variant, data.X_tr.shape[1], cfg).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    grads = GradientCollector(model) if collect_grads else None

    loss_curve: list[float] = []
    best_te, steps_to_eps, n_clipped, diverged = float("inf"), -1, 0, False
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(data.X_tr), data.y_tr)
        loss_val = loss.item()
        if not math.isfinite(loss_val) or loss_val > DIVERGENCE_LOSS:
            diverged = True
            break

        loss.backward()
        gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.clip))
        n_clipped += gnorm > cfg.clip
        opt.step()
        sched.step()
        if grads:
            grads.tick()

        if steps_to_eps < 0 and loss_val < EPS_TARGET:
            steps_to_eps = epoch
        if epoch % cfg.log_every == 0:
            loss_curve.append(loss_val)
            model.eval()
            best_te = min(best_te, _mse(model, data.X_te, data.y_te))

    if grads:
        grads.remove()

    model.eval()
    if diverged:
        nmse_tr = nmse_te = float("inf")
    else:
        nmse_tr = _mse(model, data.X_tr, data.y_tr)
        nmse_te = _mse(model, data.X_te, data.y_te)
        if not (math.isfinite(nmse_tr) and math.isfinite(nmse_te)):
            diverged, nmse_tr, nmse_te = True, float("inf"), float("inf")
        best_te = min(best_te, nmse_te)

    n_steps = max(1, epoch + 1 - diverged)
    return RunResult(
        variant=variant, target=target, seed=seed, tag=tag, config=cfg.to_dict(),
        nmse_tr=nmse_tr, nmse_te=nmse_te, best_nmse_te=best_te,
        steps_to_eps=steps_to_eps, n_params=count_params(model),
        time_sec=time.time() - t0, diverged=diverged,
        clip_rate=n_clipped / n_steps, loss_curve=loss_curve,
        grad_samples=grads.samples if grads else {},
    )
