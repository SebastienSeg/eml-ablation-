"""
figures.py — figures (matplotlib), table LaTeX et rapport Markdown.

  fig1_in_domain.pdf         NMSE médiane ± IC95, cibles in-domain
  fig2_out_of_domain.pdf     idem, cibles out-of-domain
  fig3_grad_histogram.pdf    histogramme RÉEL de ‖∂L/∂(sortie de couche)‖
  fig4_hp_robustness.pdf     heatmap NMSE EML vs (lr, δ), médiane sur init_scale
  fig5_radioactive.pdf       boxplot NMSE, décroissance radioactive
  fig6_delta_sensitivity.pdf NMSE EML vs δ
  fig7_learning_curves.pdf   loss train vs epoch (médiane ± IQR)
  table1_full.tex            tableau booktabs
  RESULTS.md                 rapport lisible généré automatiquement
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stats_analysis import GroupSummary, TestResult, bootstrap_median_ci
from training import RunResult

_CMAP = plt.get_cmap("tab10")
_FIXED = {"EML": "#1D9E75", "EML-noexp": "#BA7517", "EML-nolog": "#E3A857",
          "exp-only": "#888780", "log-only": "#5F5E5A", "ReLU-MLP": "#2B2B2A",
          "KAN": "#534AB7", "EML-v2": "#D6336C", "EML→ReLU": "#1C7ED6"}
MARKERS = {"EML": "D", "EML-noexp": "s", "EML-nolog": "s", "exp-only": "o",
           "log-only": "o", "ReLU-MLP": "^", "KAN": "p", "EML-v2": "*", "EML→ReLU": "P"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})


def color(v: str) -> str:
    return _FIXED.get(v) or matplotlib.colors.to_hex(_CMAP(sum(map(ord, v)) % 10))


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {path}")


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 & 2 — NMSE par cible
# ─────────────────────────────────────────────────────────────────────────────

def fig_nmse_grid(summaries: dict[tuple, GroupSummary], targets: list[str],
                  variants: list[str], out: Path, name: str, title: str):
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    for ax, t in zip(axes[0], targets):
        labels = []
        for i, v in enumerate(variants):
            s = summaries.get((v, t))
            if s is None:
                labels.append(v)
                continue
            labels.append(f"{v}\n({s.n_params}p)")
            if not np.isfinite(s.nmse_median):
                ax.annotate("div.", (i, 1), xycoords=("data", "axes fraction"),
                            ha="center", va="top", fontsize=6, color="red")
                continue
            lo = s.nmse_median - s.nmse_ci95_lo if np.isfinite(s.nmse_ci95_lo) else 0
            hi = s.nmse_ci95_hi - s.nmse_median if np.isfinite(s.nmse_ci95_hi) else 0
            ax.errorbar(i, s.nmse_median, yerr=[[max(lo, 0)], [max(hi, 0)]],
                        fmt=MARKERS.get(v, "o"), color=color(v), capsize=3,
                        capthick=0.8, linewidth=0.8, markersize=6)
            if s.div_rate > 0:
                ax.annotate(f"{s.div_rate:.0%}", (i, s.nmse_median), fontsize=5,
                            color="red", xytext=(4, 4), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_title(t.replace("_", " "), fontsize=8)
    axes[0][0].set_ylabel("NMSE test (médiane, IC95)")
    fig.suptitle(title, fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, out, name)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — gradients
# ─────────────────────────────────────────────────────────────────────────────

def fig_gradient_histogram(grad_runs: list[RunResult], out: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.linspace(-8, 6, 57)
    for r in grad_runs:
        vals = _finite([g for layer in r.grad_samples.values() for g in layer])
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        ax.hist(np.log10(vals), bins=bins, density=True, histtype="step",
                linewidth=1.2, color=color(r.variant),
                label=f"{r.variant} (clip {r.clip_rate:.0%})")
    ax.axvline(-6, color="#cc0000", lw=0.7, ls=":")
    ax.axvline(2, color="#e88000", lw=0.7, ls=":")
    ax.set_xlabel("log₁₀ ‖∂L/∂(sortie de couche)‖")
    ax.set_ylabel("densité")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("Gradients des couches cachées (clip = % de steps clippés)", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig3_grad_histogram.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — robustesse HP
# ─────────────────────────────────────────────────────────────────────────────

def fig_hp_robustness(sweep: list[RunResult], lr_vals, delta_vals, out: Path, target: str):
    by = defaultdict(list)
    for r in sweep:
        by[(r.config["lr"], r.config["delta"])].append(r.nmse_te)
    grid = np.full((len(lr_vals), len(delta_vals)), np.nan)
    for i, lr in enumerate(lr_vals):
        for j, d in enumerate(delta_vals):
            if by.get((lr, d)):
                grid[i, j] = np.median(by[(lr, d)])

    fig, ax = plt.subplots(figsize=(4.2, 3))
    im = ax.imshow(np.log10(grid), aspect="auto", cmap="YlOrBr")
    for i in range(len(lr_vals)):
        for j in range(len(delta_vals)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1e}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(delta_vals)), [f"δ={d:g}" for d in delta_vals], fontsize=7)
    ax.set_yticks(range(len(lr_vals)), [f"lr={lr:g}" for lr in lr_vals], fontsize=7)
    ax.set_title(f"log₁₀ NMSE — EML sur {target} (médiane sur init_scale)", fontsize=8)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    _save(fig, out, "fig4_hp_robustness.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — radioactive
# ─────────────────────────────────────────────────────────────────────────────

def fig_radioactive(results: list[RunResult], variants: list[str], out: Path):
    by = defaultdict(list)
    for r in results:
        by[r.variant].append(r.nmse_te)
    vs = [v for v in variants if v in by]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bp = ax.boxplot([_finite(by[v]) for v in vs], patch_artist=True,
                    medianprops={"color": "white", "linewidth": 1.5})
    for patch, v in zip(bp["boxes"], vs):
        patch.set_facecolor(color(v))
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(vs) + 1), vs, rotation=40, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("NMSE test")
    ax.set_title("N(t) = exp(−λt) + bruit (test sans bruit)", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig5_radioactive.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — δ
# ─────────────────────────────────────────────────────────────────────────────

def fig_delta_sensitivity(results: list[RunResult], delta_vals, out: Path, default: float):
    by = defaultdict(list)
    for r in results:
        by[r.config["delta"]].append(r.nmse_te)
    med, err_lo, err_hi = [], [], []
    for d in delta_vals:
        x = np.array(by.get(d, [np.inf]))
        m = float(np.median(x))
        lo, hi = bootstrap_median_ci(x)
        med.append(m)
        err_lo.append(max(m - lo, 0) if np.isfinite(lo) and np.isfinite(m) else 0)
        err_hi.append(max(hi - m, 0) if np.isfinite(hi) and np.isfinite(m) else 0)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.errorbar(range(len(delta_vals)), med, yerr=[err_lo, err_hi], fmt="D-",
                color=color("EML"), capsize=4, linewidth=1.2)
    ax.set_xticks(range(len(delta_vals)), [f"{d:.0e}" for d in delta_vals], fontsize=7)
    if default in delta_vals:
        ax.axvline(delta_vals.index(default), color="#999", lw=0.7, ls=":",
                   label=f"défaut ({default:g})")
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("δ dans ln(|z| + δ)")
    ax.set_ylabel("NMSE test (médiane)")
    ax.set_yscale("log")
    ax.set_title("Sensibilité de EML à δ", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig6_delta_sensitivity.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — courbes d'apprentissage
# ─────────────────────────────────────────────────────────────────────────────

def fig_learning_curves(results: list[RunResult], target: str, variants: list[str], out: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for v in variants:
        curves = [r.loss_curve for r in results
                  if r.variant == v and r.target == target and not r.diverged and r.loss_curve]
        if not curves:
            continue
        n = min(map(len, curves))
        arr = np.array([c[:n] for c in curves])
        every = results[0].config["log_every"]
        ep = np.arange(n) * every
        ax.plot(ep, np.median(arr, 0), color=color(v), lw=1.2, label=v)
        ax.fill_between(ep, *np.percentile(arr, [25, 75], 0), color=color(v), alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NMSE train")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.set_title(f"Courbes d'apprentissage — {target.replace('_', ' ')}", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig7_learning_curves.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(x: float) -> str:
    return "div." if not np.isfinite(x) else f"{x:.2e}"


def table_latex(summaries, tests: list[TestResult], targets, variants, n_seeds: int) -> str:
    tmap = {(t.other, t.target): t for t in tests}
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{NMSE test médiane (IC 95\% bootstrap), {n_seeds} seeds, "
        r"réseaux iso-paramétriques. $p$ : Mann-Whitney bilatéral vs EML, "
        r"corrigé BH ; * : significatif à 5\%.}",
        r"\label{tab:eml_ablation}", r"\small",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Cible & Variant & NMSE$_{\mathrm{med}}$ & IC95 & Conv.\,\% & Div.\,\% & $p_{\mathrm{BH}}$ \\",
        r"\midrule",
    ]
    for t in targets:
        first = True
        for v in variants:
            s = summaries.get((v, t))
            if s is None:
                continue
            tr = tmap.get((v, t))
            p = "—" if tr is None else f"{tr.p_adj:.3f}{'*' if tr.significant else ''}"
            name = rf"\textbf{{{v}}}" if v == "EML" else v
            lines.append(
                f"{t.replace('_', chr(92) + '_') if first else ''} & {name} & "
                f"{_fmt(s.nmse_median)} & [{_fmt(s.nmse_ci95_lo)}, {_fmt(s.nmse_ci95_hi)}] & "
                f"{100 * s.conv_rate:.0f}\\% & {100 * s.div_rate:.0f}\\% & {p} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def results_markdown(sections: list[tuple[str, dict, list[str]]], variants: list[str],
                     tests: list[TestResult], meta: dict) -> str:
    """Rapport Markdown : un tableau de NMSE par section + verdicts des tests."""
    md = [
        "# Résultats (généré automatiquement par `main.py`)", "",
        f"Seeds : {meta['n_seeds']} · epochs : {meta['epochs']} · "
        f"hidden réf. : {meta['hidden']} · depth : {meta['depth']} · device : {meta['device']}", "",
        "Métrique : NMSE test = MSE / Var_train(y) sur données standardisées "
        "(1.0 ≈ prédire la moyenne). Médiane sur les seeds, runs divergents comptés comme +inf. "
        "**Gras** = meilleure médiane de la ligne.", "",
    ]
    for title, summ, targets in sections:
        md += [f"## {title}", "", "| Cible | " + " | ".join(variants) + " |",
               "|---|" + "---:|" * len(variants)]
        for t in targets:
            vals = [summ.get((v, t)) for v in variants]
            meds = [s.nmse_median if s else np.inf for s in vals]
            best = int(np.argmin(meds))
            cells = []
            for i, s in enumerate(vals):
                c = "—" if s is None else _fmt(s.nmse_median)
                if s is not None and s.div_rate > 0:
                    c += f" ({s.div_rate:.0%} div)"
                cells.append(f"**{c}**" if i == best else c)
            md.append(f"| `{t}` | " + " | ".join(cells) + " |")
        md.append("")

    if tests:
        md += ["## Tests : EML vs chaque variant", "",
               "Mann-Whitney bilatéral, correction BH. r > 0 ⇔ EML meilleur. "
               "ratio = log10(médiane autre / médiane EML).", "",
               "| Cible | Variant | p (BH) | r | ratio | Verdict EML |",
               "|---|---|---:|---:|---:|---|"]
        for t in tests:
            md.append(f"| `{t.target}` | {t.other} | {t.p_adj:.3g} | {t.rank_biserial:+.2f} | "
                      f"{t.log10_ratio:+.2f} | {t.verdict} |")
        n = len(tests)
        b = sum(t.verdict == "better" for t in tests)
        w = sum(t.verdict == "worse" for t in tests)
        md += ["", f"Bilan : EML significativement meilleur dans {b}/{n} comparaisons, "
                   f"significativement pire dans {w}/{n}, non significatif dans {n - b - w}/{n}.", ""]
    return "\n".join(md)
