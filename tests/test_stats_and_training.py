import math

import numpy as np
import pytest
import torch

from config import TrainConfig
from runner import Job, load_results, run_jobs
from stats_analysis import compare, run_tests, summarize
from targets import IN_DOMAIN, OUT_OF_DOMAIN, f_cross_entropy, get_dataset
from training import RunResult, run_one


def _fake(variant, target, seed, nmse):
    return RunResult(variant, target, seed, "main", TrainConfig().to_dict(),
                     nmse, nmse, nmse, -1, 100, 0.0, not math.isfinite(nmse), 0.0)


def test_compare_direction_and_effect_size():
    better = np.array([1e-4, 2e-4, 3e-4, 1.5e-4, 2.5e-4])
    worse = better * 100
    p, r, ratio = compare(better, worse)
    assert p < 0.05 and r == pytest.approx(1.0) and ratio == pytest.approx(2.0)
    p, r, _ = compare(worse, better)
    assert p < 0.05 and r == pytest.approx(-1.0)


def test_diverged_runs_count_as_worst():
    ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    other = np.array([np.inf] * 5)
    _, r, _ = compare(ref, other)
    assert r == pytest.approx(1.0)


def test_two_sided_tests_detect_eml_being_worse():
    res = [_fake("EML", "t", s, 1.0 + s * 0.01) for s in range(8)]
    res += [_fake("ReLU-MLP", "t", s, 1e-3 * (1 + s * 0.01)) for s in range(8)]
    (t,) = run_tests(res, ["t"], "EML", ["ReLU-MLP"])
    assert t.significant and t.verdict == "worse"


def test_summary_keeps_diverged_runs():
    res = [_fake("EML", "t", s, v) for s, v in enumerate([1e-3, 2e-3, math.inf, math.inf, math.inf])]
    s = summarize(res)[("EML", "t")]
    assert s.div_rate == pytest.approx(0.6)
    assert math.isinf(s.nmse_median)       # la majorité diverge → médiane infinie


def test_cross_entropy_target_is_exact():
    x = torch.tensor([[0.0, 0.0], [20.0, -20.0]])
    y = f_cross_entropy(x)
    assert y[0].item() == pytest.approx(2 * math.log(2), rel=1e-6)
    assert y[1].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("target", [*IN_DOMAIN, *OUT_OF_DOMAIN, "radioactive_decay"])
def test_datasets_are_deterministic_and_standardized(target):
    a, b = get_dataset(target, 0), get_dataset(target, 0)
    assert torch.equal(a.X_tr, b.X_tr) and torch.equal(a.y_te, b.y_te)
    assert a.y_tr.mean().abs() < 1e-4 and abs(a.y_tr.std().item() - 1) < 1e-3


def test_run_one_learns_something():
    cfg = TrainConfig(epochs=300)
    r = run_one("ReLU-MLP", "explog_product", 0, cfg)
    assert not r.diverged and r.nmse_te < 0.5
    assert len(r.loss_curve) == 300 // cfg.log_every


def test_run_one_collects_gradients():
    r = run_one("EML", "explog_product", 0, TrainConfig(epochs=30), collect_grads=True)
    assert len(r.grad_samples) == 3          # depth couches Primitive
    assert all(len(v) == 3 for v in r.grad_samples.values())


def test_runner_persists_and_resumes(tmp_path):
    cfg = TrainConfig(epochs=5)
    jobs = [Job("main", v, "explog_product", 0, cfg) for v in ("EML", "ReLU-MLP")]
    path = tmp_path / "runs.jsonl"
    first = run_jobs(jobs, path, workers=1)
    assert len(load_results(path)) == 2
    second = run_jobs(jobs, path, workers=1)        # tout en cache : rien n'est relancé
    assert len(load_results(path)) == 2
    assert [r.nmse_te for r in first] == [r.nmse_te for r in second]
