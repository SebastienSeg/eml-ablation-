import math

import numpy as np
import pytest
import torch

from config import TrainConfig
from runner import Job, load_results, run_jobs
from stats_analysis import compare, global_verdict, run_tests, summarize
from targets import (ALL_TARGETS, FAVORABLE_FAMILIES, TARGETS, f_cross_entropy, family_of,
                     get_dataset, targets_of)
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


@pytest.mark.parametrize("target", ALL_TARGETS)
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


def test_every_family_has_targets():
    for fam in ["in-domain", "log-essential", "extrapolation", "radioactive",
                "ood", "extrapolation-control"]:
        assert targets_of(fam)


@pytest.mark.parametrize("target", targets_of("extrapolation") + targets_of("extrapolation-control"))
def test_extrapolation_domains_are_disjoint(target):
    spec = TARGETS[target]
    assert spec.test_domain is not None
    disjoint = [tr[1] <= te[0] or te[1] <= tr[0]
                for tr, te in zip(spec.bounds(False), spec.bounds(True))]
    assert any(disjoint)                 # au moins une dimension hors du domaine d'entraînement


def test_log_uniform_spans_decades():
    spec = TARGETS["log_ratio"]
    import torch as _t
    from targets import _sample
    x = _sample(spec, 5000, False, _t.Generator().manual_seed(0))
    frac_below_1 = (x[:, 0] < 1).float().mean().item()
    assert 0.45 < frac_below_1 < 0.55     # moitié des points sous 1 sur [1e-2, 1e2]


def _runs(variant, target, values):
    return [_fake(variant, target, s, v) for s, v in enumerate(values)]


def test_global_verdict_refutes_when_eml_never_wins():
    good, bad = [1e-4 * (1 + 0.01 * i) for i in range(8)], [1e-2 * (1 + 0.01 * i) for i in range(8)]
    res = []
    for t in ["explog_product", "log_ratio", "polynomial"]:
        res += _runs("EML", t, bad) + _runs("ReLU-MLP", t, good)
    gv = global_verdict(res, ["explog_product", "log_ratio", "polynomial"], family_of,
                        FAVORABLE_FAMILIES, ["EML"], ["ReLU-MLP"])
    assert gv.verdict == "réfutée" and gv.favorable_won == 0 and gv.favorable_n == 2


def test_global_verdict_requires_beating_every_comparator():
    good, mid, bad = ([k * (1 + 0.01 * i) for i in range(8)] for k in (1e-4, 1e-3, 1e-2))
    res = _runs("EML", "log_ratio", mid) + _runs("ReLU-MLP", "log_ratio", bad) \
        + _runs("exp-only", "log_ratio", good)
    gv = global_verdict(res, ["log_ratio"], family_of, FAVORABLE_FAMILIES,
                        ["EML"], ["ReLU-MLP", "exp-only"])
    assert not gv.per_target[0].won     # bat ReLU mais pas exp-only → non gagnée


def test_global_verdict_supports_when_eml_wins_favorables_only():
    good, bad = [1e-4 * (1 + 0.01 * i) for i in range(8)], [1e-2 * (1 + 0.01 * i) for i in range(8)]
    res = []
    for t in ["explog_product", "log_ratio"]:
        res += _runs("EML-v2", t, good) + _runs("ReLU-MLP", t, bad)
    res += _runs("EML-v2", "polynomial", bad) + _runs("ReLU-MLP", "polynomial", good)
    gv = global_verdict(res, ["explog_product", "log_ratio", "polynomial"], family_of,
                        FAVORABLE_FAMILIES, ["EML", "EML-v2"], ["ReLU-MLP"])
    assert gv.verdict == "soutenue" and gv.control_won == 0
