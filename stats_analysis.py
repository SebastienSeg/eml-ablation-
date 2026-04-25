"""
stats_analysis.py — agrégation et tests statistiques.

Tests utilisés :
  - Mann-Whitney U : non-paramétrique, non-paired (architectures différentes)
  - Bootstrap CI (10 000 resamplings) : intervalles de confiance robustes
  - Cohen's d : taille d'effet (> 0.8 = large, > 1.2 = très large)
  - BH correction (Benjamini-Hochberg) : contrôle du FDR sur comparaisons multiples
"""

import numpy as np
from scipy import stats as sps
from dataclasses import dataclass
from training import RunResult


# ─────────────────────────────────────────────────────────────────────────────
# Résumé agrégé par (variant, target)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroupSummary:
    variant:       str
    target:        str
    n_runs:        int
    n_diverged:    int
    mse_mean:      float
    mse_std:       float
    mse_median:    float
    mse_ci95_lo:   float   # bootstrap 2.5%
    mse_ci95_hi:   float   # bootstrap 97.5%
    n_params:      int
    param_eff:     float   # MSE_median · n_params  (bas = mieux)
    conv_rate:     float   # % runs ayant atteint EPS_TARGET
    steps_median:  int     # −1 si aucun run n'a convergé
    time_mean_sec: float
    div_rate:      float   # % runs divergents


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng  = np.random.default_rng(0)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True)
    boot_medians = np.median(boot, axis=1)
    return float(np.percentile(boot_medians, 100 * alpha / 2)), \
           float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))


def summarize(results: list[RunResult]) -> dict[tuple, GroupSummary]:
    """Retourne un dict (variant, target) → GroupSummary."""
    from collections import defaultdict
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


# ─────────────────────────────────────────────────────────────────────────────
# Mann-Whitney U + Cohen's d
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    comparison: str
    target:     str
    p_value:    float
    cohens_d:   float
    significant: bool   # après BH correction
    direction:  str     # "EML_better" | "EML_worse" | "tie"


def mann_whitney_test(eml_mses: np.ndarray, base_mses: np.ndarray) -> tuple[float, float]:
    """
    Mann-Whitney U unilatéral : H₁ = EML < baseline.
    Retourne (p_value, cohen_d).

    Pourquoi Mann-Whitney et non Wilcoxon signed-rank :
    Les variants ont des architectures différentes (hidden, couches),
    donc même seed donne des initialisations sans relation. Le seul
    couplage est sur le dataset (identique), ce qui ne suffit pas à
    justifier un test paired. Mann-Whitney U est le test non-paramétrique
    non-paired approprié.
    """
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
    """
    Compare EML à chaque baseline sur chaque target.
    Applique la correction Benjamini-Hochberg sur l'ensemble des p-values.
    """
    from primitives import VARIANTS_MAIN

    raw: list[tuple] = []  # (comparison, target, p, d)
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

    # Correction BH
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


# ─────────────────────────────────────────────────────────────────────────────
# Sweep d'hyperparamètres
# ─────────────────────────────────────────────────────────────────────────────

def analyze_hp_sweep(sweep_results: list[RunResult]) -> dict:
    """
    Pour chaque (lr, delta, init_scale), calcule la MSE médiane de EML.
    Retourne un dict hiérarchique pour visualisation heatmap.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in sweep_results:
        key = (r.variant, r.target)
        groups[key].append(r.final_mse_te)

    return {str(k): {
        "median": float(np.median(v)),
        "std":    float(np.std(v)),
        "n":      len(v),
    } for k, v in groups.items()}
