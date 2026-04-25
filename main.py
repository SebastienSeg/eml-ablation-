"""
main.py — orchestrateur de l'expérience de publication EML.

Structure :
  1. Expérience principale   : 25 seeds × 7 variants × 4 targets in-domain
  2. Out-of-domain           : 25 seeds × 7 variants × 3 targets OOD
  3. Données réelles bruitées: 25 seeds × 7 variants × décroissance radioactive
  4. Sweep hyperparamètres   : 5 seeds × 27 configs HP × target principale
  5. Ablation δ              : 5 seeds × 5 valeurs δ × EML
  6. Histogrammes de gradient: 1 seed × variants principaux
  7. Analyse statistique     : Mann-Whitney U + BH + bootstrap CI
  8. Génération des figures  : 6 figures PDF + 1 tableau LaTeX

Usage :
    pip install torch numpy scipy matplotlib tqdm
    python main.py [--quick]   # --quick réduit à 5 seeds pour test rapide
"""

import sys
import json
import time
import argparse
import itertools
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import (DEVICE, N_SEEDS_MAIN, N_SEEDS_SWEEP, HIDDEN, DEPTH,
                    N_TRAIN, N_TEST, LR_DEFAULT, DELTA_DEFAULT, HP_SWEEP,
                    DELTA_ABLATION)
from targets  import IN_DOMAIN, OUT_OF_DOMAIN, generate_radioactive_decay
from primitives import VARIANTS_MAIN, build_network, count_params
from training import run_one, collect_gradient_histogram, RunResult
from stats_analysis import summarize, run_all_tests, analyze_hp_sweep
from figures import (fig_main_results, fig_out_of_domain,
                     fig_gradient_histogram, fig_hp_robustness,
                     fig_radioactive, fig_delta_sensitivity,
                     fig_learning_curves, table_latex)


# ─────────────────────────────────────────────────────────────────────────────
# Seeds déterministes par cible (hash() est randomisé depuis Python 3.3)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SEEDS = {
    "explog_product":  101,
    "power_function":  202,
    "logistic_log":    303,
    "cross_entropy":   404,
    "polynomial":      505,
    "trigonometric":   606,
    "piecewise":       707,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"\n{'─' * w}")
    print(f"  {title}")
    print(f"{'─' * w}")


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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(n_seeds: int, run_sweep: bool = True):
    t_global = time.time()

    print(f"\n{'=' * 64}")
    print(f"  EML ABLATION STUDY — publication version")
    print(f"  Device : {DEVICE}  |  Seeds : {n_seeds}  |  Variants : {len(VARIANTS_MAIN)}")
    print(f"{'=' * 64}")

    # Vérification iso-paramétrique
    for sample_dim in sorted({spec[1] for spec in IN_DOMAIN.values()}):
        print(f"\n  Param counts (in_dim={sample_dim}, ref_hidden={HIDDEN}) :")
        for v in VARIANTS_MAIN:
            net = build_network(v, sample_dim, HIDDEN, DEPTH)
            print(f"    {v:12s}  {count_params(net):6d} trainable params")

    all_results:  list[RunResult] = []
    ood_results:  list[RunResult] = []
    rad_results:  list[RunResult] = []
    sweep_results: list[RunResult] = []

    # ── 1. Expérience principale (in-domain) ─────────────────────────────────
    print_section("1/6 — Expérience principale (in-domain)")

    for target_name, (fn, in_dim, (lo, hi)) in IN_DOMAIN.items():
        print(f"\n  Cible : {target_name}")
        # Dataset identique pour tous les variants (seed fixe par target)
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

    # ── 2. Out-of-domain ─────────────────────────────────────────────────────
    print_section("2/6 — Out-of-domain (polynomial, trig, piecewise)")

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

    # ── 3. Données réelles bruitées ───────────────────────────────────────────
    print_section("3/6 — Décroissance radioactive bruitée")

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

    # ── 4. Sweep hyperparamètres ──────────────────────────────────────────────
    if run_sweep:
        print_section("4/6 — Sweep hyperparamètres (EML uniquement)")
        target_name, (fn, in_dim, (lo, hi)) = next(iter(IN_DOMAIN.items()))
        X_tr_s, y_tr_s, X_te_s, y_te_s = make_dataset(fn, in_dim, lo, hi, 99)

        hp_combos = list(itertools.product(
            HP_SWEEP["lr"], HP_SWEEP["delta"], HP_SWEEP["init_scale"]
        ))
        print(f"  {len(hp_combos)} configurations × {N_SEEDS_SWEEP} seeds")

        for lr, delta, init_scale in tqdm(hp_combos, desc="  HP sweep", ncols=60):
            key = f"(EML, {target_name}, {lr}, {delta}, {init_scale})"
            for seed in range(N_SEEDS_SWEEP):
                r = run_one("EML", f"{target_name}_hp_{lr}_{delta}_{init_scale}",
                            seed, X_tr_s, y_tr_s, X_te_s, y_te_s,
                            lr=lr, delta=delta, init_scale=init_scale)
                sweep_results.append(r)
    else:
        print_section("4/6 — Sweep HP ignoré (mode rapide)")

    # ── 5. Ablation δ (stabilité numérique du log) ──────────────────────────
    delta_data: dict[float, list[float]] = {}
    if run_sweep:
        print_section("5/6 — Ablation δ (EML uniquement)")
        target_name_d, (fn_d, in_dim_d, (lo_d, hi_d)) = next(iter(IN_DOMAIN.items()))
        X_tr_d, y_tr_d, X_te_d, y_te_d = make_dataset(fn_d, in_dim_d, lo_d, hi_d, 99)
        print(f"  {len(DELTA_ABLATION)} valeurs de δ × {N_SEEDS_SWEEP} seeds")

        for delta_val in tqdm(DELTA_ABLATION, desc="  δ ablation", ncols=60):
            mses = []
            for seed in range(N_SEEDS_SWEEP):
                r = run_one("EML", f"{target_name_d}_delta_{delta_val}",
                            seed, X_tr_d, y_tr_d, X_te_d, y_te_d,
                            delta=delta_val)
                mses.append(r.final_mse_te)
            delta_data[delta_val] = mses
            med = float(np.median([m for m in mses if not np.isinf(m)] or [float("inf")]))
            n_div = sum(1 for m in mses if np.isinf(m))
            print(f"    δ={delta_val:.0e}  MSE={med:.5f}  div={n_div}/{N_SEEDS_SWEEP}")
    else:
        print_section("5/6 — Ablation δ ignorée (mode rapide)")

    # ── 6. Histogrammes de gradient ───────────────────────────────────────────
    print_section("6/6 — Gradients (EML vs ReLU-MLP)")

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

    # ── Analyse statistique ───────────────────────────────────────────────────
    print_section("Analyse statistique")

    summaries_in  = summarize(all_results)
    summaries_ood = summarize(ood_results)
    summaries_rad = summarize(rad_results)

    tests_in  = run_all_tests(all_results,  list(IN_DOMAIN.keys()))
    tests_ood = run_all_tests(ood_results,  list(OUT_OF_DOMAIN.keys()))

    sig_in    = [t for t in tests_in  if t.significant]
    sig_total = len(tests_in)

    print(f"\n  In-domain  : {len(sig_in)}/{sig_total} tests significatifs (BH α=0.05)")
    print(f"  Conclusion : {'combinaison causale' if len(sig_in)/sig_total > 0.75 else 'résultat mixte'}")

    for test in tests_in:
        if "noexp" in test.comparison or "nolog" in test.comparison:
            sig  = "***" if test.p_value < 0.001 else ("**" if test.p_value < 0.01 else "*")
            bh   = " [BH✓]" if test.significant else ""
            print(f"    {test.comparison:22s}  {test.target:20s}"
                  f"  p={test.p_value:.4f}{sig}  d={test.cohens_d:.2f}{bh}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print_section("Génération des figures")

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
    target_0 = next(iter(IN_DOMAIN.keys()))
    fig_learning_curves(all_results, target_0, VARIANTS_MAIN)

    latex = table_latex(summaries_in, tests_in,
                        list(IN_DOMAIN.keys()), VARIANTS_MAIN)
    (Path("figures") / "table1_full.tex").write_text(latex)
    print("  → figures/table1_full.tex")

    # ── Export JSON (résultats bruts reproductibles) ──────────────────────────
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
    print("\n  → eml_results_pub.json")

    elapsed = time.time() - t_global
    print(f"\n{'=' * 64}")
    print(f"  Terminé en {elapsed/60:.1f} min")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true",
                        help="5 seeds, pas de sweep HP (test rapide)")
    parser.add_argument("--seeds",  type=int, default=N_SEEDS_MAIN)
    args = parser.parse_args()

    n = 5 if args.quick else args.seeds
    main(n_seeds=n, run_sweep=not args.quick)
