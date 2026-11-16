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

try:
    import psutil
except ImportError:          # compteur CPU optionnel
    psutil = None

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


def default_workers() -> int:
    """Moitié des cœurs logiques : laisse la machine utilisable pendant le calcul."""
    return max(1, (os.cpu_count() or 2) // 2)


def _lower_priority():
    """Priorité basse pour le process courant : l'OS sert d'abord les autres applis."""
    try:
        if psutil is not None and os.name == "nt":
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif hasattr(os, "nice"):
            os.nice(10)
    except Exception:
        pass


def _worker(job: Job) -> RunResult:
    import torch
    torch.set_num_threads(1)     # parallélisme au niveau des runs, pas intra-op
    return run_one(job.variant, job.target, job.seed, job.cfg, tag=job.tag,
                   collect_grads=job.collect_grads)


def cpu_usage() -> str:
    """% d'utilisation CPU de toute la machine depuis le dernier appel (+ RAM)."""
    if psutil is not None:
        return f"{psutil.cpu_percent(interval=None):3.0f}% RAM {psutil.virtual_memory().percent:.0f}%"
    if hasattr(os, "getloadavg"):      # repli Unix sans psutil : charge / nb de cœurs
        return f"{100 * os.getloadavg()[0] / (os.cpu_count() or 1):3.0f}%"
    return "n/a"


class _CpuBar(tqdm):
    """Barre tqdm qui affiche l'utilisation CPU, rafraîchie au plus 1×/s."""

    def update(self, n=1):
        now = self._time()
        if now - getattr(self, "_cpu_t", 0) >= 1.0:
            self._cpu_t = now
            self.set_postfix_str(f"CPU {cpu_usage()}", refresh=False)
        return super().update(n)


def run_jobs(jobs: list[Job], results_path: Path, workers: int = 0,
             desc: str = "runs", low_priority: bool = True) -> list[RunResult]:
    """Exécute les jobs manquants, renvoie les résultats de TOUS les jobs demandés."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r.key: r for r in load_results(results_path)}
    todo = [j for j in jobs if j.key not in done]
    if len(todo) < len(jobs):
        print(f"  {len(jobs) - len(todo)}/{len(jobs)} runs déjà en cache ({results_path})")

    workers = workers or default_workers()
    if psutil is not None:
        psutil.cpu_percent(interval=None)      # amorce la mesure
    with results_path.open("a") as f:
        def save(r: RunResult):
            done[r.key] = r
            f.write(json.dumps(r.to_dict()) + "\n")
            f.flush()

        if workers == 1 or len(todo) <= 1:
            if low_priority:
                _lower_priority()
            for j in _CpuBar(todo, desc=desc, ncols=100):
                save(_worker(j))
        elif todo:
            init = _lower_priority if low_priority else None
            with ProcessPoolExecutor(max_workers=workers, initializer=init) as ex:
                futures = [ex.submit(_worker, j) for j in todo]
                for fut in _CpuBar(as_completed(futures), total=len(futures),
                                   desc=desc, ncols=100):
                    save(fut.result())

    return [done[j.key] for j in jobs]
