"""
main.py — orchestrateur de l'étude d'ablation EML.

Étapes (une par famille de cibles, cf. targets.py) :
  in-domain, log-essential, extrapolation, radioactive   (favorables à EML)
  ood, extrapolation-control                             (contrôles négatifs)
  hp-sweep   : EML × 27 configs (lr, δ, init_scale) × 5 seeds
  delta      : EML × 5 valeurs de δ × 5 seeds
  gradients  : 1 run par variant avec hooks de gradient
  analyse    : Mann-Whitney bilatéral + BH, VERDICT GLOBAL, figures, RESULTS.md

Chaque run est sauvegardé dès qu'il termine dans <out>/runs.jsonl ; relancer
la commande reprend là où elle s'était arrêtée.

Exemples :
    python main.py --quick                       # 5 seeds, sans sweeps
    python main.py                               # protocole complet (25 seeds)
    python main.py --variants EML EML-v2 ReLU-MLP --targets logistic_log
    python main.py --analyze-only                # refait stats + figures depuis le JSONL
"""

from __future__ import annotations

import argparse
import os
import itertools
import json
import time
from pathlib import Path

import numpy as np

from config import (DEVICE, DELTA_ABLATION, HP_SWEEP, N_SEEDS_MAIN, N_SEEDS_QUICK,
                    N_SEEDS_SWEEP, TrainConfig)
from primitives import VARIANTS, build_network, count_params, iso_hidden
from runner import Job, load_results, run_jobs
from primitives import variants_by_family
from stats_analysis import global_verdict, run_tests, summarize
from targets import (ALL_TARGETS, FAMILIES, FAVORABLE_FAMILIES,
                     IN_DOMAIN, RADIOACTIVE, family_of, targets_of)
import figures as F

STAGES = [*FAMILIES, "hp-sweep", "delta", "gradients"]


def section(title: str):
    print(f"\n{'─' * 64}\n  {title}\n{'─' * 64}")


def print_param_counts(variants: list[str], cfg: TrainConfig):
    for d in sorted({s.in_dim for s in IN_DOMAIN.values()}):
        print(f"\n  Paramètres entraînables (in_dim={d}, hidden réf.={cfg.hidden}) :")
        for v in variants:
            h = iso_hidden(v, d, cfg.hidden, cfg.depth)
            n = count_params(build_network(v, d, cfg))
            print(f"    {v:10s} hidden={h:3d}  params={n:5d}")


def selected_targets(args, family: str) -> list[str]:
    return [t for t in targets_of(family) if not args.targets or t in args.targets]


def build_jobs(args, cfg: TrainConfig) -> dict[str, list[Job]]:
    seeds = range(args.seeds)
    jobs: dict[str, list[Job]] = {}
    for fam in FAMILIES:
        jobs[fam] = [Job("main", v, t, s, cfg) for t in selected_targets(args, fam)
                     for v in args.variants for s in seeds]

    st = args.sweep_target
    jobs["hp-sweep"] = [
        Job("hp_sweep", "EML", st, s, cfg.with_(lr=lr, delta=d, init_scale=i))
        for lr, d, i in itertools.product(HP_SWEEP["lr"], HP_SWEEP["delta"], HP_SWEEP["init_scale"])
        for s in range(N_SEEDS_SWEEP)]
    jobs["delta"] = [Job("delta", "EML", st, s, cfg.with_(delta=d))
                     for d in DELTA_ABLATION for s in range(N_SEEDS_SWEEP)]
    jobs["gradients"] = [Job("gradients", v, st, 0, cfg, collect_grads=True)
                         for v in args.variants]
    return {k: v for k, v in jobs.items() if k in args.stages and v}


FIG_NAMES = {
    "in-domain": "fig1_in_domain.pdf", "ood": "fig2_out_of_domain.pdf",
    "log-essential": "fig8_log_essential.pdf", "extrapolation": "fig9_extrapolation.pdf",
    "extrapolation-control": "fig10_extrapolation_control.pdf",
}


def analyze(results: list, args, cfg: TrainConfig, out: Path, elapsed: float):
    section("Analyse statistique")
    main_runs = [r for r in results if r.tag == "main" and r.variant in args.variants
                 and r.config == cfg.to_dict() and r.seed < args.seeds]
    present = {r.target for r in main_runs}
    fam_targets = {f: [t for t in targets_of(f) if t in present] for f in FAMILIES}
    all_t = [t for t in ALL_TARGETS if t in present]
    summ = summarize(main_runs)
    n_seeds = max((r.seed for r in main_runs), default=-1) + 1

    tests = []
    if "EML" in args.variants:
        tests = run_tests(main_runs, all_t, reference="EML", others=args.variants)
        for fam in FAMILIES:
            sub = [t for t in tests if t.target in fam_targets[fam]]
            if sub:
                b = sum(t.verdict == "better" for t in sub)
                w = sum(t.verdict == "worse" for t in sub)
                print(f"  EML {fam:22s}: meilleur {b}/{len(sub)}, pire {w}/{len(sub)}")

    members = [v for v in variants_by_family("eml", "upgrade") if v in args.variants]
    comparators = [v for v in args.variants if v not in members]
    gv = None
    if members and comparators:
        gv = global_verdict(main_runs, all_t, family_of, FAVORABLE_FAMILIES, members, comparators)
        print(f"\n  VERDICT GLOBAL : hypothèse {gv.verdict.upper()}  "
              f"(cibles favorables gagnées {gv.favorable_won}/{gv.favorable_n}, "
              f"contrôles gagnés {gv.control_won}/{gv.control_n})")
        for v in gv.per_target:
            mark = "GAGNÉE par " + ", ".join(v.winners) if v.won else "non gagnée"
            print(f"    {v.target:20s} {v.family:22s} {v.best_member}={v.best_member_nmse:.1e}"
                  f"  vs {v.best_comparator}={v.best_comparator_nmse:.1e}  → {mark}")

    section("Figures et exports")
    figs = out / "figures"
    for fam, name in FIG_NAMES.items():
        if fam_targets[fam]:
            F.fig_nmse_grid(summ, fam_targets[fam], args.variants, figs, name,
                            f"NMSE test médiane ± IC95 — {fam}")
    if fam_targets["in-domain"]:
        F.fig_learning_curves(main_runs, fam_targets["in-domain"][0], args.variants, figs)
    rad_runs = [r for r in main_runs if r.target == RADIOACTIVE]
    if rad_runs:
        F.fig_radioactive(rad_runs, args.variants, figs)
    grad_runs = [r for r in results if r.tag == "gradients" and r.variant in args.variants]
    if grad_runs:
        F.fig_gradient_histogram(grad_runs, figs)
    sweep = [r for r in results if r.tag == "hp_sweep" and r.target == args.sweep_target]
    if sweep:
        F.fig_hp_robustness(sweep, HP_SWEEP["lr"], HP_SWEEP["delta"], figs, args.sweep_target)
    delta = [r for r in results if r.tag == "delta" and r.target == args.sweep_target]
    if delta:
        F.fig_delta_sensitivity(delta, DELTA_ABLATION, figs, default=cfg.delta)

    (figs / "table1_full.tex").write_text(
        F.table_latex(summ, tests, all_t, args.variants, n_seeds))
    print(f"  → {figs / 'table1_full.tex'}")

    meta = {"n_seeds": n_seeds, "device": DEVICE, "variants": args.variants,
            "total_time_sec": elapsed, **cfg.to_dict()}
    sections = [(f, summ, fam_targets[f]) for f in FAMILIES if fam_targets[f]]
    (out / "RESULTS.md").write_text(F.results_markdown(sections, args.variants, tests, meta, gv))
    print(f"  → {out / 'RESULTS.md'}")

    def clean(x):
        return None if isinstance(x, float) and not np.isfinite(x) else x

    export = {
        "meta": meta,
        "verdict": None if gv is None else {
            "verdict": gv.verdict, "favorable_won": gv.favorable_won,
            "favorable_n": gv.favorable_n, "control_won": gv.control_won,
            "control_n": gv.control_n, "members": gv.members, "comparators": gv.comparators,
            "per_target": [{k: clean(v) for k, v in vars(x).items()} for x in gv.per_target]},
        "summaries": [{k: clean(v) for k, v in vars(s).items()} for s in summ.values()],
        "tests": [{k: clean(v) for k, v in vars(t).items()} for t in tests],
    }
    (out / "results_summary.json").write_text(json.dumps(export, indent=2, ensure_ascii=False))
    print(f"  → {out / 'results_summary.json'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help=f"{N_SEEDS_QUICK} seeds, sans hp-sweep ni ablation δ")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--targets", nargs="+", default=None,
                   choices=ALL_TARGETS)
    p.add_argument("--stages", nargs="+", default=None, choices=STAGES)
    p.add_argument("--sweep-target", default="explog_product", choices=list(IN_DOMAIN))
    p.add_argument("--workers", type=int, default=0,
                   help="nb de runs en parallèle (0 = moitié des cœurs logiques)")
    p.add_argument("--full-priority", action="store_true",
                   help="ne pas abaisser la priorité des workers (plus rapide, PC moins réactif)")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--analyze-only", action="store_true",
                   help="ne lance aucun entraînement, analyse le JSONL existant")
    args = p.parse_args(argv)
    if args.seeds is None:
        args.seeds = N_SEEDS_QUICK if args.quick else N_SEEDS_MAIN
    if args.stages is None:
        args.stages = [s for s in STAGES if not (args.quick and s in ("hp-sweep", "delta"))]
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = TrainConfig(epochs=args.epochs)
    runs_path = args.out / "runs.jsonl"
    t0 = time.time()

    print(f"\n{'=' * 64}\n  EML ABLATION STUDY\n  device={DEVICE}  seeds={args.seeds}  "
          f"epochs={cfg.epochs}  variants={len(args.variants)}\n{'=' * 64}")
    from runner import default_workers
    print(f"  workers={args.workers or default_workers()} / {os.cpu_count()} cœurs logiques"
          f"  priorité={'normale' if args.full_priority else 'basse'}")
    print_param_counts(args.variants, cfg)

    if not args.analyze_only:
        for i, (stage, jobs) in enumerate(build_jobs(args, cfg).items(), 1):
            section(f"{i}. {stage} — {len(jobs)} runs")
            run_jobs(jobs, runs_path, workers=args.workers, desc=stage,
                     low_priority=not args.full_priority)

    results = load_results(runs_path)
    if not results:
        raise SystemExit(f"aucun résultat dans {runs_path}")
    analyze(results, args, cfg, args.out, time.time() - t0)
    print(f"\n  Terminé en {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    main()
