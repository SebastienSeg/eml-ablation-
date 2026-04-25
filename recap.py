"""
recap.py — fichier unique regroupant tout le code du projet EML Ablation Study.

Concatene : config, targets, primitives, training, stats_analysis, figures, main.
Executable directement : python recap.py [--quick] [--seeds N]
"""

import sys
import json
import time
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import stats as sps
from tqdm import tqdm


# =====================================================================
# CONFIG
# =====================================================================

N_SEEDS_MAIN   = 25
N_SEEDS_SWEEP  = 5
GLOBAL_SEED    = 42

N_TRAIN        = 2000
N_TEST         = 500
N_NOISE_POINTS = 300

EPOCHS         = 3000
LR_DEFAULT     = 3e-3
CLIP_VAL       = 10.0
EPS_TARGET     = 1e-3

HIDDEN         = 24       # taille hidden de reference (pour ReLU-MLP)
DEPTH          = 3
# build_network() ajuste hidden par variant pour iso-parametrie avec ReLU-MLP

DELTA_DEFAULT  = 1e-4
EXP_CLAMP      = 6.0

HP_SWEEP = {
    "lr":         [1e-3, 3e-3, 1e-2],
    "delta":      [1e-5, 1e-4, 1e-3],
    "init_scale": [0.1,  0.5,  1.0],
}

DELTA_ABLATION = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


# =====================================================================
# TARGETS
# =====================================================================

def f_explog_product(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x[:, 0]**2) * torch.log1p(x[:, 1]**2)

def f_power(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x[:, 0] * torch.log1p(torch.abs(x[:, 1])))

def f_logistic_log(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x[:, 0]) * torch.exp(x[:, 1]) * torch.log(x[:, 2] + 2)

def f_cross_entropy(x: torch.Tensor) -> torch.Tensor:
    delta = DELTA_DEFAULT
    p = torch.sigmoid(x[:, 0])
    q = torch.sigmoid(x[:, 1])
    hp = -(p * torch.log(p + delta) + (1 - p) * torch.log(1 - p + delta))
    hq = -(q * torch.log(q + delta) + (1 - q) * torch.log(1 - q + delta))
    return hp + hq

def f_polynomial(x: torch.Tensor) -> torch.Tensor:
    u, v = x[:, 0], x[:, 1]
    return u**3 + 2 * u**2 * v - 0.5 * v**3 + 0.3 * u * v

def f_trigonometric(x: torch.Tensor) -> torch.Tensor:
    return (torch.sin(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1])
            + 0.5 * torch.sin(2 * torch.pi * x[:, 0]))

def f_piecewise(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x[:, 0]) * torch.sign(x[:, 1]) + torch.clamp(x[:, 0] * x[:, 1], -1, 1)

def generate_radioactive_decay(n: int = 300, noise_std: float = 0.05,
                               seed: int = 0) -> tuple:
    rng  = np.random.default_rng(seed)
    N0   = 1.0
    t    = torch.from_numpy(rng.uniform(0, 5, n)).float()
    lam  = torch.from_numpy(rng.uniform(0.2, 2.0, n)).float()
    noise = torch.from_numpy(rng.normal(0, noise_std, n)).float()
    y    = N0 * torch.exp(-lam * t) + noise
    X    = torch.stack([t, lam], dim=1)
    return X, y.unsqueeze(1)

IN_DOMAIN: dict = {
    "explog_product": (f_explog_product, 2, (-2.0,  2.0)),
    "power_function":  (f_power,          2, ( 0.1,  2.0)),
    "logistic_log":    (f_logistic_log,   3, (-2.0,  2.0)),
    "cross_entropy":   (f_cross_entropy,  2, (-3.0,  3.0)),
}

OUT_OF_DOMAIN: dict = {
    "polynomial":    (f_polynomial,   2, (-2.0,  2.0)),
    "trigonometric": (f_trigonometric, 2, (-1.0,  1.0)),
    "piecewise":     (f_piecewise,    2, (-2.0,  2.0)),
}


# =====================================================================
# PRIMITIVES
# =====================================================================

class EMLLayer(nn.Module):
    """EML(x) = alpha * exp(W_exp * x) - beta * ln(|W_log * x| + delta)"""

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


class ExpOnlyLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, **_):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha  = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.exp(
            torch.clamp(self.linear(x), -EXP_CLAMP, EXP_CLAMP)
        )


class LogOnlyLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 delta: float = DELTA_DEFAULT, **_):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha  = nn.Parameter(torch.ones(out_features))
        self.delta  = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.log(torch.abs(self.linear(x)) + self.delta)


class ReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, **_):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


class BSplineActivation(nn.Module):
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
    """Couche KAN vectorisee : une B-spline apprise par connexion (i -> j)."""

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
        x_exp  = x.repeat(1, self.out_features)
        x_norm = self.scale * (x_exp - self.shift)

        grid = self.grid
        xc  = x_norm.clamp(grid[0], grid[-1] - 1e-6)
        idx = torch.searchsorted(
            grid.contiguous(), xc.contiguous()
        ).clamp(1, self.k - 1)

        t = (xc - grid[idx - 1]) / (grid[idx] - grid[idx - 1] + 1e-8)

        ctrl_exp = self.ctrl.unsqueeze(0).expand(B, -1, -1)
        ctrl_lo  = ctrl_exp.gather(2, (idx - 1).unsqueeze(2)).squeeze(2)
        ctrl_hi  = ctrl_exp.gather(2, idx.unsqueeze(2)).squeeze(2)

        y = ctrl_lo * (1 - t) + ctrl_hi * t
        return y.view(B, self.out_features, self.in_features).sum(dim=2)


def _eml_frozen_exp(i, o, **kw):
    return EMLLayer(i, o, freeze_exp=True, **kw)

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
VARIANTS_DISPLAY = VARIANTS_MAIN


# -- Iso-parametrie -------------------------------------------------------

def _estimate_layer_params(variant: str, d_in: int, d_out: int) -> int:
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
    dims = [in_dim] + [hidden] * depth + [1]
    total = 0
    for i in range(len(dims) - 1):
        total += _estimate_layer_params(variant, dims[i], dims[i + 1])
        if i < len(dims) - 2:
            total += 2 * dims[i + 1]
    return total


def _find_iso_hidden(variant: str, in_dim: int,
                     ref_hidden: int, depth: int) -> int:
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


# =====================================================================
# TRAINING
# =====================================================================

@dataclass
class RunResult:
    variant:      str
    target:       str
    seed:         int
    final_mse_tr: float
    final_mse_te: float
    steps_to_eps: int
    n_params:     int
    time_sec:     float
    diverged:     bool
    loss_curve:   list = field(default_factory=list)
    grad_norms:   list = field(default_factory=list)


class GradientCollector:
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
            "pct_vanishing":  float((arr < 1e-6).mean()),
            "pct_exploding":  float((arr > 1e2).mean()),
        }


def run_one(variant: str, target: str, seed: int,
            X_tr: torch.Tensor, y_tr: torch.Tensor,
            X_te: torch.Tensor, y_te: torch.Tensor,
            lr: float = LR_DEFAULT,
            delta: float = 1e-4,
            init_scale: float = 1.0,
            collect_grads: bool = False) -> RunResult:
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


def collect_gradient_histogram(variant: str, target: str,
                                X_tr, y_tr, X_te, y_te,
                                seed: int = 0) -> dict:
    torch.manual_seed(seed)
    in_dim  = X_tr.shape[1]
    model   = build_network(variant, in_dim, HIDDEN, DEPTH).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR_DEFAULT)
    loss_fn = nn.MSELoss()
    col     = GradientCollector(model)

    for epoch in range(min(EPOCHS, 500)):
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


# =====================================================================
# STATS ANALYSIS
# =====================================================================

@dataclass
class GroupSummary:
    variant:       str
    target:        str
    n_runs:        int
    n_diverged:    int
    mse_mean:      float
    mse_std:       float
    mse_median:    float
    mse_ci95_lo:   float
    mse_ci95_hi:   float
    n_params:      int
    param_eff:     float
    conv_rate:     float
    steps_median:  int
    time_mean_sec: float
    div_rate:      float


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng  = np.random.default_rng(0)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True)
    boot_medians = np.median(boot, axis=1)
    return float(np.percentile(boot_medians, 100 * alpha / 2)), \
           float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))


def summarize(results: list[RunResult]) -> dict[tuple, GroupSummary]:
    groups: dict[tuple, list[RunResult]] = defaultdict(list)
    for r in results:
        groups[(r.variant, r.target)].append(r)

    summaries = {}
    for (v, t), runs in groups.items():
        mses      = np.array([r.final_mse_te for r in runs])
        valid     = mses[~np.isinf(mses) & ~np.isnan(mses)]
        diverged  = sum(r.diverged for r in runs)
        stps      = [r.steps_to_eps for r in runs if r.steps_to_eps != -1]
        times     = [r.time_sec for r in runs]

        if len(valid) == 0:
            valid = np.array([float("inf")])

        lo, hi = bootstrap_ci(valid) if len(valid) >= 3 else (float("nan"), float("nan"))

        summaries[(v, t)] = GroupSummary(
            variant=v, target=t, n_runs=len(runs), n_diverged=diverged,
            mse_mean=float(valid.mean()), mse_std=float(valid.std()),
            mse_median=float(np.median(valid)),
            mse_ci95_lo=lo, mse_ci95_hi=hi,
            n_params=runs[0].n_params,
            param_eff=float(np.median(valid)) * runs[0].n_params,
            conv_rate=len(stps) / len(runs),
            steps_median=int(np.median(stps)) if stps else -1,
            time_mean_sec=float(np.mean(times)),
            div_rate=diverged / len(runs),
        )
    return summaries


@dataclass
class TestResult:
    comparison: str
    target:     str
    p_value:    float
    cohens_d:   float
    significant: bool
    direction:  str


def mann_whitney_test(eml_mses: np.ndarray, base_mses: np.ndarray) -> tuple[float, float]:
    if len(eml_mses) < 3:
        return 1.0, 0.0

    e = eml_mses[~np.isinf(eml_mses)]
    b = base_mses[~np.isinf(base_mses)]
    if len(e) < 3 or len(b) < 3 or np.allclose(e, b):
        return 1.0, 0.0

    try:
        _, p = sps.mannwhitneyu(e, b, alternative="less")
    except Exception:
        p = 1.0

    pooled_std = np.sqrt((e.std()**2 + b.std()**2) / 2 + 1e-12)
    d          = (b.mean() - e.mean()) / pooled_std
    return float(p), float(d)


def run_all_tests(results: list[RunResult],
                  targets: list[str]) -> list[TestResult]:
    raw: list[tuple] = []
    for t in targets:
        eml_mses = np.array([r.final_mse_te for r in results
                             if r.variant == "EML" and r.target == t])
        for b in VARIANTS_MAIN:
            if b == "EML":
                continue
            base_mses = np.array([r.final_mse_te for r in results
                                  if r.variant == b and r.target == t])
            p, d = mann_whitney_test(eml_mses, base_mses)
            raw.append((f"EML < {b}", t, p, d))

    p_values = np.array([x[2] for x in raw])
    n        = len(p_values)
    order    = np.argsort(p_values)
    bh_threshold = (np.arange(1, n + 1) / n) * 0.05
    sig_mask = np.zeros(n, dtype=bool)
    for k in range(n - 1, -1, -1):
        if p_values[order[k]] <= bh_threshold[k]:
            sig_mask[order[:k + 1]] = True
            break

    out = []
    for i, (comp, t, p, d) in enumerate(raw):
        direction = "EML_better" if d > 0 else ("EML_worse" if d < -0.2 else "tie")
        out.append(TestResult(
            comparison=comp, target=t, p_value=p, cohens_d=d,
            significant=bool(sig_mask[i]), direction=direction
        ))
    return out


def analyze_hp_sweep(sweep_results: list[RunResult]) -> dict:
    groups = defaultdict(list)
    for r in sweep_results:
        key = (r.variant, r.target)
        groups[key].append(r.final_mse_te)

    return {str(k): {
        "median": float(np.median(v)),
        "std":    float(np.std(v)),
        "n":      len(v),
    } for k, v in groups.items()}


# =====================================================================
# FIGURES
# =====================================================================

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

PALETTE = {
    "EML":       "#1D9E75",
    "EML-noexp": "#BA7517",
    "EML-nolog": "#C07830",
    "exp-only":  "#888780",
    "log-only":  "#73726c",
    "ReLU-MLP":  "#4A4A48",
    "KAN":       "#534AB7",
}
MARKERS = {
    "EML": "D", "EML-noexp": "s", "EML-nolog": "s",
    "exp-only": "o", "log-only": "o", "ReLU-MLP": "^", "KAN": "p",
}

STYLE = {
    "font.family": "serif",
    "font.size":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def fig_main_results(summaries: dict[tuple, GroupSummary],
                     in_domain_targets: list[str],
                     variants: list[str]):
    _apply_style()
    n_targets = len(in_domain_targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(3.2 * n_targets, 3.5),
                              sharey=False)
    if n_targets == 1:
        axes = [axes]

    for ax, t in zip(axes, in_domain_targets):
        xs = np.arange(len(variants))
        for i, v in enumerate(variants):
            key = (v, t)
            if key not in summaries:
                continue
            s = summaries[key]
            med  = s.mse_median
            lo   = med - s.mse_ci95_lo
            hi   = s.mse_ci95_hi - med
            err  = np.array([[max(lo, 0)], [max(hi, 0)]])
            col  = PALETTE.get(v, "#888")
            mark = MARKERS.get(v, "o")
            ax.errorbar(i, med, yerr=err, fmt=mark, color=col,
                        capsize=3, capthick=0.8, linewidth=0.8,
                        markersize=6, label=v)

        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax.set_title(t.replace("_", " "), fontsize=8, pad=4)
        ax.set_ylabel("MSE test (log)" if ax == axes[0] else "")

    handles = [plt.Line2D([0], [0], marker=MARKERS.get(v, "o"),
                          color=PALETTE.get(v, "#888"), linewidth=0,
                          markersize=5, label=v)
               for v in variants]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("MSE mediane +/- IC95% bootstrap -- fonctions in-domain",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    path = OUT / "fig1_main_results.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def fig_out_of_domain(summaries: dict[tuple, GroupSummary],
                      ood_targets: list[str],
                      variants: list[str]):
    _apply_style()
    n = len(ood_targets)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, ood_targets):
        for i, v in enumerate(variants):
            key = (v, t)
            if key not in summaries:
                continue
            s   = summaries[key]
            col = PALETTE.get(v, "#888")
            ls  = "--" if v == "EML" else "-"
            ax.plot(i, s.mse_median, marker=MARKERS.get(v, "o"),
                    color=col, linestyle=ls, markersize=6)

        ax.set_yscale("log")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax.set_title(t.replace("_", " "), fontsize=8, pad=4)
        ax.set_ylabel("MSE test (log)" if ax == axes[0] else "")

    fig.suptitle("MSE mediane -- fonctions out-of-domain (EML en pointille)",
                 fontsize=9, y=1.01)
    note = "EML ne beneficie d'aucun avantage structurel sur les fonctions polynomiales/trig."
    fig.text(0.5, -0.06, note, ha="center", fontsize=7, color="#666",
             style="italic")
    fig.tight_layout()
    path = OUT / "fig2_out_of_domain.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def fig_gradient_histogram(grad_data: dict[str, dict]):
    _apply_style()
    fig, ax = plt.subplots(figsize=(5, 3))

    for v, g in grad_data.items():
        if not g:
            continue
        mu    = g.get("mean", 0)
        sigma = g.get("std",  1e-8)
        if mu <= 0 or sigma <= 0:
            continue
        rng     = np.random.default_rng(0)
        samples = rng.lognormal(np.log(mu + 1e-12),
                                np.log(1 + sigma / (mu + 1e-12)), 1000)
        ax.hist(np.log10(samples + 1e-12), bins=40, alpha=0.55,
                label=f"{v}  (vanish={g['pct_vanishing']:.1%})",
                color=PALETTE.get(v, "#888"), density=True, linewidth=0)

    ax.axvline(-6, color="#cc0000", linewidth=0.7, linestyle=":",
               label="seuil vanishing (1e-6)")
    ax.axvline(2,  color="#e88000", linewidth=0.7, linestyle=":",
               label="seuil exploding (1e2)")
    ax.set_xlabel("log10(||dloss/dlayer_output||)")
    ax.set_ylabel("densite")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Distribution des normes de gradient des sorties de couche", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig3_grad_histogram.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def fig_hp_robustness(hp_grid: dict, lr_vals: list, delta_vals: list,
                      init_vals: list, target: str = "explog_product"):
    _apply_style()
    grid = np.zeros((len(lr_vals), len(delta_vals)))
    for i, lr in enumerate(lr_vals):
        for j, d in enumerate(delta_vals):
            key = f"('EML', '{target}', {lr}, {d})"
            if key in hp_grid:
                grid[i, j] = hp_grid[key]["median"]
            else:
                grid[i, j] = float("nan")

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(np.log10(grid + 1e-12), aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(len(delta_vals)))
    ax.set_xticklabels([f"d={d}" for d in delta_vals], fontsize=7)
    ax.set_yticks(range(len(lr_vals)))
    ax.set_yticklabels([f"lr={lr}" for lr in lr_vals], fontsize=7)
    ax.set_title(f"log10(MSE) -- EML sur {target}", fontsize=8)
    plt.colorbar(im, ax=ax, label="log10(MSE)")
    fig.tight_layout()
    path = OUT / "fig4_hp_robustness.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def fig_radioactive(results_radioactive: list[RunResult]):
    _apply_style()
    groups = defaultdict(list)
    for r in results_radioactive:
        groups[r.variant].append(r.final_mse_te)

    variants = list(groups.keys())
    data     = [np.array(groups[v]) for v in variants]
    colors   = [PALETTE.get(v, "#888") for v in variants]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 1.5})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(0.7)
    for cap in bp["caps"]:
        cap.set_linewidth(0.7)
    for flier in bp["fliers"]:
        flier.set(marker="o", markersize=3, alpha=0.4)

    ax.set_xticks(range(1, len(variants) + 1))
    ax.set_xticklabels(variants, rotation=40, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("MSE test")
    ax.set_title("Fit N(t) = N0*exp(-lt) + bruit gaussien", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig5_radioactive.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def fig_delta_sensitivity(delta_results: dict[float, list[float]],
                          delta_vals: list[float]):
    _apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    medians, ci_los, ci_his = [], [], []
    for d in delta_vals:
        mses = np.array(delta_results[d])
        valid = mses[~np.isinf(mses) & ~np.isnan(mses)]
        if len(valid) >= 3:
            med = float(np.median(valid))
            lo, hi = bootstrap_ci(valid)
        else:
            med, lo, hi = float("inf"), float("nan"), float("nan")
        medians.append(med)
        ci_los.append(med - lo)
        ci_his.append(hi - med)

    err = np.array([
        [max(l, 0) for l in ci_los],
        [max(h, 0) for h in ci_his],
    ])
    ax.errorbar(range(len(delta_vals)), medians, yerr=err,
                fmt="D-", color=PALETTE["EML"], capsize=4, capthick=0.8,
                linewidth=1.2, markersize=6)

    ax.set_xticks(range(len(delta_vals)))
    ax.set_xticklabels([f"{d:.0e}" for d in delta_vals], fontsize=7)
    ax.set_xlabel("delta (terme de stabilite du log)")
    ax.set_ylabel("MSE test (mediane)")
    ax.set_yscale("log")
    ax.axvline(delta_vals.index(1e-4), color="#999", linewidth=0.7,
               linestyle=":", label="delta par defaut (1e-4)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Sensibilite de EML au parametre delta", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig6_delta_sensitivity.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def table_latex(summaries: dict[tuple, GroupSummary],
                tests: list[TestResult],
                targets: list[str], variants: list[str]) -> str:
    sig_map = {tr.comparison + "|" + tr.target: tr for tr in tests}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{R\'esultats d'ablation EML. MSE test m\'ediane (IC 95\% bootstrap),"
        r" 25 seeds, architecture iso-param\'etrique.}",
        r"\label{tab:eml_ablation}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Cible & Variant & MSE$_{\text{med}}$ & IC95 & Conv\% & Div\% & t (s) & $p$ \\",
        r"\midrule",
    ]

    for t in targets:
        first = True
        for v in variants:
            key = (v, t)
            if key not in summaries:
                continue
            s    = summaries[key]
            t_key = f"EML < {v}|{t}"
            tr   = sig_map.get(t_key)
            p_str = f"{tr.p_value:.3f}{'*' if tr and tr.significant else ''}" if tr else "---"

            target_cell = t.replace("_", r"\_") if first else ""
            bold_s = r"\textbf" if v == "EML" else ""

            row = (f"{target_cell} & {bold_s}{{{v}}} & "
                   f"{bold_s}{{{s.mse_median:.4f}}} & "
                   f"[{s.mse_ci95_lo:.4f},{s.mse_ci95_hi:.4f}] & "
                   f"{s.conv_rate:.0%} & "
                   f"{s.div_rate:.0%} & "
                   f"{s.time_mean_sec:.1f} & "
                   f"{p_str} \\\\")
            lines.append(row)
            first = False
        lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================

TARGET_SEEDS = {
    "explog_product":  101,
    "power_function":  202,
    "logistic_log":    303,
    "cross_entropy":   404,
    "polynomial":      505,
    "trigonometric":   606,
    "piecewise":       707,
}


def make_dataset(fn, in_dim, lo, hi, seed):
    torch.manual_seed(seed + 10_000)
    X_tr = torch.FloatTensor(N_TRAIN, in_dim).uniform_(lo, hi).to(DEVICE)
    X_te = torch.FloatTensor(N_TEST,  in_dim).uniform_(lo, hi).to(DEVICE)
    with torch.no_grad():
        y_tr = fn(X_tr).unsqueeze(1)
        y_te = fn(X_te).unsqueeze(1)
    return X_tr, y_tr, X_te, y_te


def print_section(title: str):
    w = 64
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def divergence_report(results: list[RunResult], label: str):
    total   = len(results)
    n_div   = sum(r.diverged for r in results)
    by_var  = {}
    for r in results:
        by_var.setdefault(r.variant, []).append(r.diverged)
    print(f"\n  Divergences ({label}) : {n_div}/{total} runs")
    for v, divs in by_var.items():
        rate = sum(divs) / len(divs)
        if rate > 0:
            print(f"    {v:12s}  {rate:.0%} divergent")


def main(n_seeds: int, run_sweep: bool = True):
    t_global = time.time()

    print(f"\n{'=' * 64}")
    print(f"  EML ABLATION STUDY -- publication version")
    print(f"  Device : {DEVICE}  |  Seeds : {n_seeds}  |  Variants : {len(VARIANTS_MAIN)}")
    print(f"{'=' * 64}")

    # Verification iso-parametrique
    for sample_dim in sorted({spec[1] for spec in IN_DOMAIN.values()}):
        print(f"\n  Param counts (in_dim={sample_dim}, ref_hidden={HIDDEN}) :")
        for v in VARIANTS_MAIN:
            net = build_network(v, sample_dim, HIDDEN, DEPTH)
            print(f"    {v:12s}  {count_params(net):6d} trainable params")

    all_results:  list[RunResult] = []
    ood_results:  list[RunResult] = []
    rad_results:  list[RunResult] = []
    sweep_results: list[RunResult] = []

    # -- 1. Experience principale (in-domain) ---------------------------------
    print_section("1/6 -- Experience principale (in-domain)")

    for target_name, (fn, in_dim, (lo, hi)) in IN_DOMAIN.items():
        print(f"\n  Cible : {target_name}")
        base_seed = TARGET_SEEDS[target_name]
        X_tr, y_tr, X_te, y_te = make_dataset(fn, in_dim, lo, hi, base_seed)

        for variant in VARIANTS_MAIN:
            mses = []
            for seed in tqdm(range(n_seeds), desc=f"    {variant:12s}",
                             leave=False, ncols=60):
                r = run_one(variant, target_name, seed,
                            X_tr, y_tr, X_te, y_te)
                all_results.append(r)
                mses.append(r.final_mse_te if not r.diverged else float("inf"))

            median_mse = float(np.median([m for m in mses if not np.isinf(m)] or [float("inf")]))
            div_count  = sum(r.diverged for r in all_results
                             if r.variant == variant and r.target == target_name)
            print(f"    {variant:12s}  MSE={median_mse:.5f}  div={div_count}/{n_seeds}")

    divergence_report(all_results, "in-domain")

    # -- 2. Out-of-domain -----------------------------------------------------
    print_section("2/6 -- Out-of-domain (polynomial, trig, piecewise)")

    for target_name, (fn, in_dim, (lo, hi)) in OUT_OF_DOMAIN.items():
        print(f"\n  Cible OOD : {target_name}")
        base_seed = TARGET_SEEDS[target_name] + 1000
        X_tr, y_tr, X_te, y_te = make_dataset(fn, in_dim, lo, hi, base_seed)

        for variant in VARIANTS_MAIN:
            mses = []
            for seed in tqdm(range(n_seeds), desc=f"    {variant:12s}",
                             leave=False, ncols=60):
                r = run_one(variant, target_name, seed,
                            X_tr, y_tr, X_te, y_te)
                ood_results.append(r)
                mses.append(r.final_mse_te if not r.diverged else float("inf"))

            median_mse = float(np.median([m for m in mses if not np.isinf(m)] or [float("inf")]))
            print(f"    {variant:12s}  MSE={median_mse:.5f}")

    divergence_report(ood_results, "out-of-domain")

    # -- 3. Donnees reelles bruitees -------------------------------------------
    print_section("3/6 -- Decroissance radioactive bruitee")

    for seed in tqdm(range(n_seeds), desc="  Radioactive", ncols=60):
        X_tr_r, y_tr_r = generate_radioactive_decay(N_TRAIN, seed=seed)
        X_te_r, y_te_r = generate_radioactive_decay(N_TEST,  seed=seed + 5000)
        X_tr_r, y_tr_r = X_tr_r.to(DEVICE), y_tr_r.to(DEVICE)
        X_te_r, y_te_r = X_te_r.to(DEVICE), y_te_r.to(DEVICE)

        for variant in VARIANTS_MAIN:
            r = run_one(variant, "radioactive_decay", seed,
                        X_tr_r, y_tr_r, X_te_r, y_te_r)
            rad_results.append(r)

    for variant in VARIANTS_MAIN:
        sub   = [r for r in rad_results if r.variant == variant]
        mses  = [r.final_mse_te for r in sub if not r.diverged]
        print(f"  {variant:12s}  MSE={np.median(mses):.5f}" if mses else f"  {variant:12s}  (diverged)")

    # -- 4. Sweep hyperparametres ----------------------------------------------
    if run_sweep:
        print_section("4/6 -- Sweep hyperparametres (EML uniquement)")
        target_name, (fn, in_dim, (lo, hi)) = next(iter(IN_DOMAIN.items()))
        X_tr_s, y_tr_s, X_te_s, y_te_s = make_dataset(fn, in_dim, lo, hi, 99)

        hp_combos = list(itertools.product(
            HP_SWEEP["lr"], HP_SWEEP["delta"], HP_SWEEP["init_scale"]
        ))
        print(f"  {len(hp_combos)} configurations x {N_SEEDS_SWEEP} seeds")

        for lr, delta, init_scale in tqdm(hp_combos, desc="  HP sweep", ncols=60):
            key = f"(EML, {target_name}, {lr}, {delta}, {init_scale})"
            for seed in range(N_SEEDS_SWEEP):
                r = run_one("EML", f"{target_name}_hp_{lr}_{delta}_{init_scale}",
                            seed, X_tr_s, y_tr_s, X_te_s, y_te_s,
                            lr=lr, delta=delta, init_scale=init_scale)
                sweep_results.append(r)
    else:
        print_section("4/6 -- Sweep HP ignore (mode rapide)")

    # -- 5. Ablation delta -----------------------------------------------------
    delta_data: dict[float, list[float]] = {}
    if run_sweep:
        print_section("5/6 -- Ablation delta (EML uniquement)")
        target_name_d, (fn_d, in_dim_d, (lo_d, hi_d)) = next(iter(IN_DOMAIN.items()))
        X_tr_d, y_tr_d, X_te_d, y_te_d = make_dataset(fn_d, in_dim_d, lo_d, hi_d, 99)
        print(f"  {len(DELTA_ABLATION)} valeurs de delta x {N_SEEDS_SWEEP} seeds")

        for delta_val in tqdm(DELTA_ABLATION, desc="  delta ablation", ncols=60):
            mses = []
            for seed in range(N_SEEDS_SWEEP):
                r = run_one("EML", f"{target_name_d}_delta_{delta_val}",
                            seed, X_tr_d, y_tr_d, X_te_d, y_te_d,
                            delta=delta_val)
                mses.append(r.final_mse_te)
            delta_data[delta_val] = mses
            med = float(np.median([m for m in mses if not np.isinf(m)] or [float("inf")]))
            n_div = sum(1 for m in mses if np.isinf(m))
            print(f"    delta={delta_val:.0e}  MSE={med:.5f}  div={n_div}/{N_SEEDS_SWEEP}")
    else:
        print_section("5/6 -- Ablation delta ignoree (mode rapide)")

    # -- 6. Histogrammes de gradient -------------------------------------------
    print_section("6/6 -- Gradients (EML vs ReLU-MLP)")

    grad_data = {}
    target_name_0, (fn_0, in_dim_0, (lo_0, hi_0)) = next(iter(IN_DOMAIN.items()))
    X_tr_g, y_tr_g, X_te_g, y_te_g = make_dataset(fn_0, in_dim_0, lo_0, hi_0, 7)

    for v in ["EML", "EML-noexp", "EML-nolog", "ReLU-MLP", "KAN"]:
        grad_data[v] = collect_gradient_histogram(
            v, target_name_0, X_tr_g, y_tr_g, X_te_g, y_te_g, seed=0
        )
        g = grad_data[v]
        if g:
            print(f"  {v:12s}  grad_mean={g['mean']:.4f}"
                  f"  vanish={g['pct_vanishing']:.1%}"
                  f"  explode={g['pct_exploding']:.1%}")

    # -- Analyse statistique ---------------------------------------------------
    print_section("Analyse statistique")

    summaries_in  = summarize(all_results)
    summaries_ood = summarize(ood_results)
    summaries_rad = summarize(rad_results)

    tests_in  = run_all_tests(all_results,  list(IN_DOMAIN.keys()))
    tests_ood = run_all_tests(ood_results,  list(OUT_OF_DOMAIN.keys()))

    sig_in    = [t for t in tests_in  if t.significant]
    sig_total = len(tests_in)

    print(f"\n  In-domain  : {len(sig_in)}/{sig_total} tests significatifs (BH a=0.05)")
    print(f"  Conclusion : {'combinaison causale' if len(sig_in)/sig_total > 0.75 else 'resultat mixte'}")

    for test in tests_in:
        if "noexp" in test.comparison or "nolog" in test.comparison:
            sig  = "***" if test.p_value < 0.001 else ("**" if test.p_value < 0.01 else "*")
            bh   = " [BH ok]" if test.significant else ""
            print(f"    {test.comparison:22s}  {test.target:20s}"
                  f"  p={test.p_value:.4f}{sig}  d={test.cohens_d:.2f}{bh}")

    # -- Figures ---------------------------------------------------------------
    print_section("Generation des figures")

    fig_main_results(summaries_in, list(IN_DOMAIN.keys()), VARIANTS_MAIN)
    fig_out_of_domain(summaries_ood, list(OUT_OF_DOMAIN.keys()), VARIANTS_MAIN)
    fig_gradient_histogram(grad_data)
    if run_sweep and sweep_results:
        hp_analysis = analyze_hp_sweep(sweep_results)
        fig_hp_robustness(hp_analysis, HP_SWEEP["lr"], HP_SWEEP["delta"],
                          HP_SWEEP["init_scale"])
    fig_radioactive(rad_results)
    if run_sweep and delta_data:
        fig_delta_sensitivity(delta_data, DELTA_ABLATION)

    latex = table_latex(summaries_in, tests_in,
                        list(IN_DOMAIN.keys()), VARIANTS_MAIN)
    (Path("figures") / "table1_full.tex").write_text(latex)
    print("  -> figures/table1_full.tex")

    # -- Export JSON -----------------------------------------------------------
    export = {
        "meta": {
            "n_seeds":  n_seeds,
            "hidden":   HIDDEN,
            "depth":    DEPTH,
            "n_train":  N_TRAIN,
            "n_test":   N_TEST,
            "device":   DEVICE,
            "total_time_sec": time.time() - t_global,
        },
        "in_domain": {
            f"{v}|{t}": {
                "mse_median": s.mse_median,
                "mse_ci95":   [s.mse_ci95_lo, s.mse_ci95_hi],
                "conv_rate":  s.conv_rate,
                "div_rate":   s.div_rate,
                "time_mean":  s.time_mean_sec,
                "n_params":   s.n_params,
            }
            for (v, t), s in summaries_in.items()
        },
        "out_of_domain": {
            f"{v}|{t}": {"mse_median": s.mse_median, "div_rate": s.div_rate}
            for (v, t), s in summaries_ood.items()
        },
        "radioactive": {
            f"{v}|radioactive_decay": {"mse_median": s.mse_median}
            for (v, t), s in summaries_rad.items()
        },
        "gradient_stats": grad_data,
        "statistical_tests": [
            {"comparison": tr.comparison, "target": tr.target,
             "p_value": tr.p_value, "cohens_d": tr.cohens_d,
             "significant_BH": tr.significant}
            for tr in tests_in
        ],
    }

    Path("eml_results_pub.json").write_text(json.dumps(export, indent=2))
    print("\n  -> eml_results_pub.json")

    elapsed = time.time() - t_global
    print(f"\n{'=' * 64}")
    print(f"  Termine en {elapsed/60:.1f} min")
    print(f"{'=' * 64}\n")


# =====================================================================
# SANITY CHECK KAN
# =====================================================================

def _sanity_check_kan():
    """Verifie que KANLayer vectorise == ancienne boucle Python."""

    class KANLayer_loop(nn.Module):
        def __init__(self, in_f, out_f, k=8, x_min=-6.0, x_max=6.0, **_):
            super().__init__()
            self.in_features, self.out_features = in_f, out_f
            self.activations = nn.ModuleList([
                BSplineActivation(k, x_min, x_max)
                for _ in range(in_f * out_f)
            ])

        def forward(self, x):
            B = x.shape[0]
            out = torch.zeros(B, self.out_features, device=x.device)
            for j in range(self.out_features):
                for i in range(self.in_features):
                    act = self.activations[j * self.in_features + i]
                    out[:, j] = out[:, j] + act(x[:, i])
            return out

    for in_f, out_f in [(2, 3), (3, 4), (1, 1), (5, 2)]:
        torch.manual_seed(42)
        ref = KANLayer_loop(in_f, out_f)
        vec = KANLayer(in_f, out_f)
        with torch.no_grad():
            for a, act in enumerate(ref.activations):
                vec.ctrl[a]  = act.ctrl.data
                vec.scale[a] = act.scale.data.squeeze()
                vec.shift[a] = act.shift.data.squeeze()

        for B in [1, 8, 64]:
            torch.manual_seed(7 + B)
            x = torch.randn(B, in_f)
            assert torch.allclose(ref(x), vec(x), atol=1e-6), \
                f"KAN mismatch ({in_f}->{out_f}, B={B})"

    print("  KAN vectorization: OK")


if __name__ == "__main__":
    _sanity_check_kan()

    parser = argparse.ArgumentParser(description="EML Ablation Study")
    parser.add_argument("--quick",  action="store_true",
                        help="5 seeds, pas de sweep HP (test rapide)")
    parser.add_argument("--seeds",  type=int, default=N_SEEDS_MAIN)
    args = parser.parse_args()

    n = 5 if args.quick else args.seeds
    main(n_seeds=n, run_sweep=not args.quick)
