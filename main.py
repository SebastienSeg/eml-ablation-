"""
main.py — orchestrateur de l'étude d'ablation EML.

Étapes :
  1. in-domain      : variants × 4 cibles exp/log × N seeds
  2. out-of-domain  : variants × 3 cibles sans structure exp/log × N seeds
  3. radioactive    : variants × décroissance bruitée × N seeds
  4. hp-sweep       : EML × 27 configs (lr, δ, init_scale) × 5 seeds
  5. delta          : EML × 5 valeurs de δ × 5 seeds
  6. gradients      : 1 run par variant avec hooks de gradient
  7. analyse        : Mann-Whitney bilatéral + BH, figures, LaTeX, RESULTS.md, JSON

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
import itertools
import json
import time
from pathlib import Path

import numpy as np

from config import (DEVICE, DELTA_ABLATION, HP_SWEEP, N_SEEDS_MAIN, N_SEEDS_QUICK,
                    N_SEEDS_SWEEP, TrainConfig)
from primitives import VARIANTS, build_network, count_params, iso_hidden
from runner import Job, load_results, run_jobs
from stats_analysis import run_tests, summarize
from targets import IN_DOMAIN, OUT_OF_DOMAIN, RADIOACTIVE
import figures as F

STAGES = ["in-domain", "ood", "radioactive", "hp-sweep", "delta", "gradients"]


def section(title: str):
    print(f"\n{'─' * 64}\n  {title}\n{'─' * 64}")


def print_param_counts(variants: list[str], cfg: TrainConfig):
    for d in sorted({s.in_dim for s in IN_DOMAIN.values()}):
        print(f"\n  Paramètres entraînables (in_dim={d}, hidden réf.={cfg.hidden}) :")
        for v in variants:
            h = iso_hidden(v, d, cfg.hidden, cfg.depth)
            n = count_params(build_network(v, d, cfg))
            print(f"    {v:10s} hidden={h:3d}  params={n:5d}")


def build_jobs(args, cfg: TrainConfig) -> dict[str, list[Job]]:
    seeds = range(args.seeds)
    in_t  = [t for t in IN_DOMAIN if not args.targets or t in args.targets]
    ood_t = [t for t in OUT_OF_DOMAIN if not args.targets or t in args.targets]
    jobs: dict[str, list[Job]] = {}

    jobs["in-domain"] = [Job("main", v, t, s, cfg)
                         for t in in_t for v in args.variants for s in seeds]
    jobs["ood"] = [Job("main", v, t, s, cfg)
                   for t in ood_t for v in args.variants for s in seeds]
    if not args.targets or RADIOACTIVE in args.targets:
        jobs["radioactive"] = [Job("main", v, RADIOACTIVE, s, cfg)
                               for v in args.variants for s in seeds]

    sweep_target = args.sweep_target
    if "EML" in VARIANTS:
        jobs["hp-sweep"] = [
            Job("hp_sweep", "EML", sweep_target, s, cfg.with_(lr=lr, delta=d, init_scale=i))
            for lr, d, i in itertools.product(HP_SWEEP["lr"], HP_SWEEP["delta"],
                                              HP_SWEEP["init_scale"])
            for s in range(N_SEEDS_SWEEP)]
        jobs["delta"] = [Job("delta", "EML", sweep_target, s, cfg.with_(delta=d))
                         for d in DELTA_ABLATION for s in range(N_SEEDS_SWEEP)]
    jobs["gradients"] = [Job("gradients", v, sweep_target, 0, cfg, collect_grads=True)
                         for v in args.variants]
    return {k: v for k, v in jobs.items() if k in args.stages and v}


def analyze(results: list, args, cfg: TrainConfig, out: Path, elapsed: float):
    section("Analyse statistique")
    main_runs = [r for r in results if r.tag == "main" and r.variant in args.variants
                 and r.config == cfg.to_dict()]
    in_t  = [t for t in IN_DOMAIN if any(r.target == t for r in main_runs)]
    ood_t = [t for t in OUT_OF_DOMAIN if any(r.target == t for r in main_runs)]
    rad_runs = [r for r in main_runs if r.target == RADIOACTIVE]
    summ = summarize(main_runs)
    n_seeds = max((r.seed for r in main_runs), default=-1) + 1

    tests = []
    if "EML" in args.variants:
        tests = run_tests(main_runs, in_t + ood_t + ([RADIOACTIVE] if rad_runs else []),
                          reference="EML", others=args.variants)
        for fam, ts in (("in-domain", in_t), ("out-of-domain", ood_t)):
            sub = [t for t in tests if t.target in ts]
            if sub:
                b = sum(t.verdict == "better" for t in sub)
                w = sum(t.verdict == "worse" for t in sub)
                print(f"  {fam:14s}: EML meilleur {b}/{len(sub)}, pire {w}/{len(sub)} (BH 5 %)")

    for t in in_t + ood_t + ([RADIOACTIVE] if rad_runs else []):
        ranked = sorted((s.nmse_median, v) for (v, tt), s in summ.items() if tt == t)
        print(f"  {t:18s} " + "  ".join(f"{v}={m:.1e}" for m, v in ranked[:4]))

    section("Figures et exports")
    figs = out / "figures"
    if in_t:
        F.fig_nmse_grid(summ, in_t, args.variants, figs, "fig1_in_domain.pdf",
                        "NMSE test médiane ± IC95 — cibles in-domain")
        F.fig_learning_curves(main_runs, in_t[0], args.variants, figs)
    if ood_t:
        F.fig_nmse_grid(summ, ood_t, args.variants, figs, "fig2_out_of_domain.pdf",
                        "NMSE test médiane ± IC95 — cibles out-of-domain")
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
        F.table_latex(summ, tests, in_t, args.variants, n_seeds))
    print(f"  → {figs / 'table1_full.tex'}")

    meta = {"n_seeds": n_seeds, "device": DEVICE, "variants": args.variants,
            "total_time_sec": elapsed, **cfg.to_dict()}
    sections = [(n, summ, ts) for n, ts in (("In-domain", in_t), ("Out-of-domain", ood_t),
                                            ("Décroissance radioactive", [RADIOACTIVE] if rad_runs else []))
                if ts]
    (out / "RESULTS.md").write_text(F.results_markdown(sections, args.variants, tests, meta))
    print(f"  → {out / 'RESULTS.md'}")

    def clean(x):
        return None if isinstance(x, float) and not np.isfinite(x) else x

    export = {
        "meta": meta,
        "summaries": [{k: clean(v) for k, v in vars(s).items()} for s in summ.values()],
        "tests": [{**{k: clean(v) for k, v in vars(t).items()}} for t in tests],
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
                   choices=[*IN_DOMAIN, *OUT_OF_DOMAIN, RADIOACTIVE])
    p.add_argument("--stages", nargs="+", default=None, choices=STAGES)
    p.add_argument("--sweep-target", default="explog_product", choices=list(IN_DOMAIN))
    p.add_argument("--workers", type=int, default=0, help="0 = nb de CPU")
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
    print_param_counts(args.variants, cfg)

    if not args.analyze_only:
        for i, (stage, jobs) in enumerate(build_jobs(args, cfg).items(), 1):
            section(f"{i}. {stage} — {len(jobs)} runs")
            run_jobs(jobs, runs_path, workers=args.workers, desc=stage)

    results = load_results(runs_path)
    if not results:
        raise SystemExit(f"aucun résultat dans {runs_path}")
    analyze(results, args, cfg, args.out, time.time() - t0)
    print(f"\n  Terminé en {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    main()
