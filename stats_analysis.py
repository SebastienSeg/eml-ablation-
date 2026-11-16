"""
stats_analysis.py — agrégation et tests statistiques.

Choix :
  - Les runs divergents comptent comme NMSE = +inf (pire rang) au lieu d'être
    exclus : les exclure favoriserait les variants instables (biais du survivant).
  - Mann-Whitney U BILATÉRAL (un test unilatéral « EML < baseline » ne peut
    jamais conclure qu'EML est significativement pire).
  - Taille d'effet de rang (rank-biserial r ∈ [−1, 1], > 0 ⇔ la référence a
    une NMSE plus basse) + rapport de médianes en log10. Le d de Cohen sur des
    MSE (distribution très asymétrique, sur plusieurs ordres de grandeur) n'a
    pas de sens et donnait des d > 30.
  - Correction de Benjamini-Hochberg par famille de tests (scipy).
  - IC95 bootstrap de la médiane.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from training import RunResult


# ─────────────────────────────────────────────────────────────────────────────
# Résumés
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroupSummary:
    variant:      str
    target:       str
    n_runs:       int
    n_params:     int
    nmse_median:  float
    nmse_ci95_lo: float
    nmse_ci95_hi: float
    nmse_q25:     float
    nmse_q75:     float
    div_rate:     float
    conv_rate:    float          # fraction des runs avec loss train < EPS_TARGET
    steps_median: int            # parmi les runs convergés, −1 sinon
    clip_rate:    float          # fraction moyenne de steps clippés
    time_mean:    float


def bootstrap_median_ci(values: np.ndarray, n_boot: int = 10_000,
                        alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(values, size=(n_boot, len(values))), axis=1)
    return (float(np.quantile(meds, alpha / 2, method="lower")),
            float(np.quantile(meds, 1 - alpha / 2, method="higher")))


def summarize(results: list[RunResult]) -> dict[tuple[str, str], GroupSummary]:
    groups: dict[tuple, list[RunResult]] = defaultdict(list)
    for r in results:
        groups[(r.variant, r.target)].append(r)

    out = {}
    for (v, t), runs in groups.items():
        x = np.array([r.nmse_te for r in runs])
        lo, hi = bootstrap_median_ci(x)
        steps = [r.steps_to_eps for r in runs if r.steps_to_eps >= 0]
        out[(v, t)] = GroupSummary(
            variant=v, target=t, n_runs=len(runs), n_params=runs[0].n_params,
            nmse_median=float(np.median(x)), nmse_ci95_lo=lo, nmse_ci95_hi=hi,
            nmse_q25=float(np.quantile(x, 0.25, method="lower")),
            nmse_q75=float(np.quantile(x, 0.75, method="higher")),
            div_rate=float(np.mean([r.diverged for r in runs])),
            conv_rate=len(steps) / len(runs),
            steps_median=int(np.median(steps)) if steps else -1,
            clip_rate=float(np.mean([r.clip_rate for r in runs])),
            time_mean=float(np.mean([r.time_sec for r in runs])),
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    reference:   str
    other:       str
    target:      str
    p_value:     float           # Mann-Whitney bilatéral, brut
    p_adj:       float           # après Benjamini-Hochberg
    rank_biserial: float         # > 0 ⇔ reference meilleure
    log10_ratio: float           # log10(médiane other / médiane reference)
    significant: bool            # p_adj < alpha
    verdict:     str             # "better" | "worse" | "n.s."

    @property
    def comparison(self) -> str:
        return f"{self.reference} vs {self.other}"


def compare(ref: np.ndarray, other: np.ndarray) -> tuple[float, float, float]:
    """(p bilatéral, rank-biserial, log10 ratio des médianes)."""
    n1, n2 = len(ref), len(other)
    if n1 < 2 or n2 < 2:
        return 1.0, 0.0, 0.0
    # U1 = #paires (ref > other) + ½ #égalités → P(ref < other) = 1 − U1/(n1·n2)
    u1, p = sps.mannwhitneyu(ref, other, alternative="two-sided")
    r = 1.0 - 2.0 * u1 / (n1 * n2)
    if np.all(ref == ref[0]) and np.all(other == ref[0]):
        p = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(np.log10(np.median(other) / np.median(ref)))
    return float(p), float(r), ratio


def run_tests(results: list[RunResult], targets: list[str], reference: str,
              others: list[str], alpha: float = 0.05) -> list[TestResult]:
    """Compare `reference` à chaque variant de `others` sur chaque cible ; BH sur la famille."""
    by = defaultdict(list)
    for r in results:
        by[(r.variant, r.target)].append(r.nmse_te)

    raw = []
    for t in targets:
        ref = np.array(by.get((reference, t), []))
        for o in others:
            if o == reference or (o, t) not in by or len(ref) == 0:
                continue
            raw.append((o, t, *compare(ref, np.array(by[(o, t)]))))
    if not raw:
        return []

    p_adj = sps.false_discovery_control([x[2] for x in raw], method="bh")
    out = []
    for (o, t, p, r, ratio), pa in zip(raw, p_adj):
        sig = bool(pa < alpha)
        verdict = ("better" if r > 0 else "worse") if sig else "n.s."
        out.append(TestResult(reference, o, t, p, float(pa), r, ratio, sig, verdict))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sweeps
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sweep(results: list[RunResult], keys: tuple[str, ...]) -> dict[tuple, dict]:
    """Médiane de NMSE test regroupée par les hyperparamètres `keys`."""
    groups = defaultdict(list)
    for r in results:
        groups[tuple(r.config[k] for k in keys)].append(r.nmse_te)
    return {k: {"median": float(np.median(v)), "values": v,
                "div_rate": float(np.mean(np.isinf(v)))}
            for k, v in groups.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Verdict global (règle de décision fixée AVANT de voir les résultats)
# ─────────────────────────────────────────────────────────────────────────────

REFUTE_BELOW = 0.25   # famille EML gagne < 25 % des cibles favorables → réfutée
SUPPORT_FROM = 0.50   # ≥ 50 % des favorables ET plus que sur les contrôles → soutenue


@dataclass
class TargetVerdict:
    target:          str
    family:          str
    favorable:       bool
    best_member:     str          # meilleure médiane dans la famille EML
    best_member_nmse: float
    best_comparator: str          # meilleure médiane parmi les comparateurs
    best_comparator_nmse: float
    winners:         list         # membres significativement meilleurs que TOUS les comparateurs
    losers:          list         # membres significativement pires que le meilleur comparateur

    @property
    def won(self) -> bool:
        return bool(self.winners)


@dataclass
class GlobalVerdict:
    verdict:        str           # "réfutée" | "soutenue" | "non concluante"
    favorable_won:  int
    favorable_n:    int
    control_won:    int
    control_n:      int
    per_target:     list
    members:        list
    comparators:    list

    @property
    def favorable_rate(self) -> float:
        return self.favorable_won / max(1, self.favorable_n)

    @property
    def control_rate(self) -> float:
        return self.control_won / max(1, self.control_n)


def global_verdict(results: list[RunResult], targets: list[str], family_of,
                   favorable_families: list[str], members: list[str],
                   comparators: list[str], alpha: float = 0.05) -> GlobalVerdict:
    """
    Hypothèse H : « une couche contenant exp ET log en dur apprend mieux, à
    paramètres égaux, les fonctions à structure exp/log ».

    Une cible est GAGNÉE si au moins un membre de la famille EML (EML ou un
    upgrade) est significativement meilleur que CHAQUE comparateur (baselines
    ET ablations, donc aussi exp-only et EML-nolog : gagner grâce à la seule
    exponentielle ne compte pas). Toutes les p-values (membres × comparateurs ×
    cibles) sont corrigées ensemble par BH.

    Ce critère est volontairement indulgent pour EML (il suffit d'un membre sur
    plusieurs) : un échec est donc une réfutation forte.
    """
    by = defaultdict(list)
    for r in results:
        by[(r.variant, r.target)].append(r.nmse_te)

    raw = []   # (target, member, comparator, p, r)
    for t in targets:
        for m in members:
            if (m, t) not in by:
                continue
            for c in comparators:
                if (c, t) in by:
                    p, r, _ = compare(np.array(by[(m, t)]), np.array(by[(c, t)]))
                    raw.append((t, m, c, p, r))
    p_adj = sps.false_discovery_control([x[3] for x in raw], method="bh") if raw else []
    sig = defaultdict(dict)    # (t, m) → {c: +1 meilleur / −1 pire / 0 n.s.}
    for (t, m, c, _, r), pa in zip(raw, p_adj):
        sig[(t, m)][c] = (1 if r > 0 else -1) if pa < alpha else 0

    per_target = []
    for t in targets:
        med = {v: float(np.median(by[(v, t)])) for v in members + comparators if (v, t) in by}
        mem = [m for m in members if m in med]
        com = [c for c in comparators if c in med]
        if not mem or not com:
            continue
        bm = min(mem, key=med.get)
        bc = min(com, key=med.get)
        winners = [m for m in mem if sig[(t, m)] and all(sig[(t, m)].get(c) == 1 for c in com)]
        losers = [m for m in mem if sig[(t, m)].get(bc) == -1]
        per_target.append(TargetVerdict(t, family_of(t), family_of(t) in favorable_families,
                                        bm, med[bm], bc, med[bc], winners, losers))

    fav = [v for v in per_target if v.favorable]
    ctl = [v for v in per_target if not v.favorable]
    fw, cw = sum(v.won for v in fav), sum(v.won for v in ctl)
    f_rate = fw / max(1, len(fav))
    c_rate = cw / max(1, len(ctl))
    if f_rate < REFUTE_BELOW:
        verdict = "réfutée"
    elif f_rate >= SUPPORT_FROM and f_rate > c_rate:
        verdict = "soutenue"
    else:
        verdict = "non concluante"
    return GlobalVerdict(verdict, fw, len(fav), cw, len(ctl), per_target, members, comparators)
