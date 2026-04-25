"""
figures.py — génération des figures de publication (matplotlib).

Figures produites :
  fig1_main_results.pdf     — MSE médiane ± CI95 (in-domain)
  fig2_out_of_domain.pdf    — tableau out-of-domain : EML perd (honnêteté)
  fig3_grad_histogram.pdf   — histogramme ∂loss/∂(sortie de couche) par variant
  fig4_hp_robustness.pdf    — heatmap MSE vs (LR, δ) pour EML
  fig5_radioactive.pdf      — fit décroissance radioactive bruitée
  fig6_delta_sensitivity.pdf — MSE EML vs δ ∈ [1e-6, 1e-2]
  fig7_learning_curves.pdf  — loss vs epoch (mediane ± IQR, par variant)
  table1_full.tex           — tableau LaTeX complet pour article
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from training import RunResult
from stats_analysis import GroupSummary, TestResult

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── Palette cohérente ─────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Résultats principaux (in-domain)
# ─────────────────────────────────────────────────────────────────────────────

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
        labels = []
        for v in variants:
            key = (v, t)
            if key in summaries:
                labels.append(f"{v}\n({summaries[key].n_params}p)")
            else:
                labels.append(v)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_title(t.replace("_", " "), fontsize=8, pad=4)
        ax.set_ylabel("MSE test (log)" if ax == axes[0] else "")

    # Légende commune
    handles = [plt.Line2D([0], [0], marker=MARKERS.get(v, "o"),
                          color=PALETTE.get(v, "#888"), linewidth=0,
                          markersize=5, label=v)
               for v in variants]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("MSE médiane ± IC95% bootstrap — fonctions in-domain",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    path = OUT / "fig1_main_results.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Out-of-domain (EML perd)
# ─────────────────────────────────────────────────────────────────────────────

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
            # EML en pointillé pour montrer qu'il n'est plus le meilleur
            ls  = "--" if v == "EML" else "-"
            ax.plot(i, s.mse_median, marker=MARKERS.get(v, "o"),
                    color=col, linestyle=ls, markersize=6)

        ax.set_yscale("log")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax.set_title(t.replace("_", " "), fontsize=8, pad=4)
        ax.set_ylabel("MSE test (log)" if ax == axes[0] else "")

    fig.suptitle("MSE médiane — fonctions out-of-domain (EML en pointillé)",
                 fontsize=9, y=1.01)
    note = "EML ne bénéficie d'aucun avantage structurel sur les fonctions polynomiales/trig."
    fig.text(0.5, -0.06, note, ha="center", fontsize=7, color="#666",
             style="italic")
    fig.tight_layout()
    path = OUT / "fig2_out_of_domain.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Histogramme des gradients
# ─────────────────────────────────────────────────────────────────────────────

def fig_gradient_histogram(grad_data: dict[str, dict]):
    """
    grad_data : {variant: GradientCollector.summary()}
    Affiche ∂loss/∂(sortie de couche) en log-échelle pour détecter vanishing/exploding.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(5, 3))

    # On reconstruit une distribution approchée à partir des stats
    for v, g in grad_data.items():
        if not g:
            continue
        # Approximation log-normale à partir de mean et std
        mu    = g.get("mean", 0)
        sigma = g.get("std",  1e-8)
        if mu <= 0 or sigma <= 0:
            continue
        # Simulation de 1000 points pour l'histogramme approximatif
        rng     = np.random.default_rng(0)
        samples = rng.lognormal(np.log(mu + 1e-12),
                                np.log(1 + sigma / (mu + 1e-12)), 1000)
        ax.hist(np.log10(samples + 1e-12), bins=40, alpha=0.55,
                label=f"{v}  (vanish={g['pct_vanishing']:.1%})",
                color=PALETTE.get(v, "#888"), density=True, linewidth=0)

    ax.axvline(-6, color="#cc0000", linewidth=0.7, linestyle=":",
               label="seuil vanishing (10⁻⁶)")
    ax.axvline(2,  color="#e88000", linewidth=0.7, linestyle=":",
               label="seuil exploding (10²)")
    ax.set_xlabel("log₁₀(||∂loss/∂layer_output||)")
    ax.set_ylabel("densité")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Distribution des normes de gradient des sorties de couche", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig3_grad_histogram.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Robustesse hyperparamètres
# ─────────────────────────────────────────────────────────────────────────────

def fig_hp_robustness(hp_grid: dict, lr_vals: list, delta_vals: list,
                      init_vals: list, target: str = "explog_product"):
    """
    Heatmap MSE vs (LR, δ) pour EML, moyennée sur init_scale.
    """
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
    ax.set_xticklabels([f"δ={d}" for d in delta_vals], fontsize=7)
    ax.set_yticks(range(len(lr_vals)))
    ax.set_yticklabels([f"lr={lr}" for lr in lr_vals], fontsize=7)
    ax.set_title(f"log₁₀(MSE) — EML sur {target}", fontsize=8)
    plt.colorbar(im, ax=ax, label="log₁₀(MSE)")
    fig.tight_layout()
    path = OUT / "fig4_hp_robustness.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Données réelles bruitées (décroissance radioactive)
# ─────────────────────────────────────────────────────────────────────────────

def fig_radioactive(results_radioactive: list[RunResult]):
    """
    Boxplot MSE test sur la tâche de fit de décroissance radioactive.
    """
    _apply_style()
    from collections import defaultdict
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
    ax.set_title("Fit N(t) = N₀·exp(−λt) + bruit gaussien", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig5_radioactive.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Sensibilité à δ (stabilité numérique du log)
# ─────────────────────────────────────────────────────────────────────────────

def fig_delta_sensitivity(delta_results: dict[float, list[float]],
                          delta_vals: list[float]):
    """
    delta_results : {delta_value: [mse_test_seed_0, mse_test_seed_1, ...]}
    Affiche MSE médiane ± IC95 bootstrap en fonction de δ.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    from stats_analysis import bootstrap_ci

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
    ax.set_xlabel("δ (terme de stabilité du log)")
    ax.set_ylabel("MSE test (médiane)")
    ax.set_yscale("log")
    ax.axvline(delta_vals.index(1e-4), color="#999", linewidth=0.7,
               linestyle=":", label="δ par défaut (1e-4)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Sensibilité de EML au paramètre δ", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig6_delta_sensitivity.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Courbes d'apprentissage
# ─────────────────────────────────────────────────────────────────────────────

def fig_learning_curves(results: list[RunResult],
                        target: str,
                        variants: list[str]):
    """
    Affiche loss vs epoch (mediane sur les seeds) pour un target donne.
    Utilise RunResult.loss_curve (echantillonne tous les 50 epochs).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(5, 3.2))

    for v in variants:
        curves = [r.loss_curve for r in results
                  if r.variant == v and r.target == target
                  and not r.diverged and r.loss_curve]
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        median = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        epochs = np.arange(min_len) * 50

        col = PALETTE.get(v, "#888")
        ax.plot(epochs, median, color=col, linewidth=1.2, label=v)
        ax.fill_between(epochs, p25, p75, color=col, alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss train (log)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title(f"Courbes d'apprentissage -- {target.replace('_', ' ')}",
                 fontsize=9)
    fig.tight_layout()
    path = OUT / "fig7_learning_curves.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def table_latex(summaries: dict[tuple, GroupSummary],
                tests: list[TestResult],
                targets: list[str], variants: list[str]) -> str:
    """
    Génère le code LaTeX pour table1_full.tex.
    Format : target | variant | MSE med (CI) | conv% | div% | time | p-val | sig
    """
    sig_map = {tr.comparison + "|" + tr.target: tr for tr in tests}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Résultats d'ablation EML. MSE test médiane (IC 95\% bootstrap),"
        r" 25 seeds, architecture iso-paramétrique.}",
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
            p_str = f"{tr.p_value:.3f}{'*' if tr and tr.significant else ''}" if tr else "—"

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
