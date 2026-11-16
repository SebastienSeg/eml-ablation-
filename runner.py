"""
runner.py — exécution parallèle des runs avec persistance et reprise.

Chaque run terminé est ajouté immédiatement à un fichier JSONL. Relancer
la même commande saute les runs déjà présents : une expérience de plusieurs
heures interrompue reprend là où elle s'était arrêtée, et l'analyse
statistique peut être refaite sans réentraîner (`main.py --analyze-only`).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from config import TrainConfig
from training import RunResult, run_key, run_one


@dataclass(frozen=True)
class Job:
    tag:     str
    variant: str
    target:  str
    seed:    int
    cfg:     TrainConfig
    collect_grads: bool = False

    @property
    def key(self) -> tuple:
        return run_key(self.tag, self.variant, self.target, self.seed, self.cfg)


def load_results(path: Path) -> list[RunResult]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(RunResult.from_dict(json.loads(line)))
    return out


def _worker(job: Job) -> RunResult:
    import torch
    torch.set_num_threads(1)     # parallélisme au niveau des runs, pas intra-op
    return run_one(job.variant, job.target, job.seed, job.cfg, tag=job.tag,
                   collect_grads=job.collect_grads)


def run_jobs(jobs: list[Job], results_path: Path, workers: int = 0,
             desc: str = "runs") -> list[RunResult]:
    """Exécute les jobs manquants, renvoie les résultats de TOUS les jobs demandés."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r.key: r for r in load_results(results_path)}
    todo = [j for j in jobs if j.key not in done]
    if len(todo) < len(jobs):
        print(f"  {len(jobs) - len(todo)}/{len(jobs)} runs déjà en cache ({results_path})")

    workers = workers or max(1, (os.cpu_count() or 1))
    with results_path.open("a") as f:
        def save(r: RunResult):
            done[r.key] = r
            f.write(json.dumps(r.to_dict()) + "\n")
            f.flush()

        if workers == 1 or len(todo) <= 1:
            for j in tqdm(todo, desc=desc, ncols=80):
                save(_worker(j))
        elif todo:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_worker, j) for j in todo]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=desc, ncols=80):
                    save(fut.result())

    return [done[j.key] for j in jobs]
