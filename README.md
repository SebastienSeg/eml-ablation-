# EML — Exp-Minus-Log Neural Networks

Banc d'ablation pour une question précise : **une couche neuronale qui contient
`exp` et `log` en dur apprend-elle mieux qu'un MLP classique, à nombre de
paramètres égal, les fonctions qui ont une structure exp/log ?**

```
EML(x) = α ⊙ exp(W_e·x) − β ⊙ ln(|W_l·x| + δ) + b
```

Le dépôt entraîne plusieurs familles de réseaux (EML, ablations d'EML,
baselines, « upgrades » d'EML) sur des fonctions cibles 2D/3D, puis compare
les erreurs avec des tests statistiques. Il est conçu pour pouvoir **réfuter**
l'hypothèse aussi facilement que la confirmer.

Les résultats du dernier run sont dans [`results/RESULTS.md`](results/RESULTS.md)
(fichier généré, jamais recopié à la main).

## Ce que disent les résultats (run `--quick`, 5 seeds)

En l'état, **l'hypothèse n'est pas confirmée**. L'EML d'origine est
significativement moins bon que ReLU-MLP, KAN et exp-only sur 3 des 4 cibles
in-domain, avec des NMSE 10 à 100 fois plus élevées. Il ne gagne que sur
`logistic_log`, contre ReLU-MLP et KAN, mais à égalité avec exp-only et
EML-nolog. Les ablations désignent le coupable : **retirer la branche log
(`EML-nolog`) améliore EML partout**, alors que retirer la branche exp le
détruit. La branche `ln(|z| + δ)` fait clipper 11 % des steps d'EML (77 % pour
log-only) et dégrade l'apprentissage.

L'upgrade `EML-v2` (log lisse + biais) supprime le clipping (0 %) et devient
le meilleur ou quasi-meilleur variant sur `explog_product`, `power_function`,
`cross_entropy` et `trigonometric`. Mais il fait jeu égal avec `exp-only`, ce
qui ne permet pas d'attribuer le gain à la combinaison exp + log plutôt qu'à
l'exponentielle seule. Avec 5 seeds, ces conclusions restent indicatives : le
protocole complet (`python main.py`) est nécessaire pour les confirmer.

---

## Ce que fait le code, de bout en bout

```
targets.py      cibles f(x) + génération des datasets (standardisés)
primitives.py   couches (EML, exp-only, log-only, ReLU, KAN…) + REGISTRE des variants
                + construction iso-paramétrique des réseaux
training.py     run_one() : entraîne 1 réseau, renvoie un RunResult
runner.py       exécute N runs en parallèle, les écrit dans runs.jsonl, reprend si interrompu
stats_analysis  médianes, IC bootstrap, Mann-Whitney bilatéral, correction BH
figures.py      7 figures PDF, table LaTeX, RESULTS.md
main.py         CLI qui enchaîne tout
tests/          pytest (formules, iso-paramétrie, KAN, stats, reprise…)
```

**Un réseau**, quel que soit le variant :

```
x ─▶ [Primitive → LayerNorm] × 3 ─▶ Linear(hidden → 1) ─▶ ŷ
```

Seule la primitive des couches cachées change. La largeur `hidden` est choisie
par variant pour que le nombre de paramètres entraînables soit le plus proche
possible de celui de ReLU-MLP à largeur 24 (≈ 1 450 paramètres, écart < 10 %,
vérifié par les tests).

**Un run** = (variant, cible, seed, `TrainConfig`). Entraînement full-batch,
Adam (lr 3e-3), cosine annealing, 3 000 epochs, `clip_grad_norm_ = 10`.
X et y sont standardisés avec les statistiques du train ; la métrique est donc
une **NMSE** (MSE / variance de y) : 1.0 ≈ prédire la moyenne, et les cibles
d'échelles différentes deviennent comparables.

**Une expérience** (`main.py`) :

| Étape                   | Contenu                                                             |
|-------------------------|---------------------------------------------------------------------|
| `in-domain`             | 4 cibles exp/log sur petit domaine                                  |
| `log-essential`         | 3 cibles où le log est indispensable (plusieurs décades, produits)  |
| `extrapolation`         | 4 cibles entraînées sur un domaine, testées sur un domaine disjoint |
| `radioactive`           | `exp(−λt)` + bruit (test sans bruit)                                |
| `ood`                   | 3 cibles sans exp/log (contrôle négatif)                            |
| `extrapolation-control` | extrapolation d'un polynôme (contrôle négatif)                      |
| `hp-sweep`              | EML × 27 configs (lr, δ, init_scale) × 5 seeds                      |
| `delta`                 | EML × δ ∈ {1e-6 … 1e-2} × 5 seeds                                   |
| `gradients`             | 1 run par variant avec hooks ‖∂L/∂sortie‖ par couche                |
| analyse                 | tests, **verdict global**, figures, `RESULTS.md`, JSON              |

Chaque famille de cibles est entraînée avec tous les variants × N seeds.

## Variants

| Variant     | Famille  | Couche cachée                                                     |
|-------------|----------|-------------------------------------------------------------------|
| `EML`       | eml      | `α·exp(W_e·x) − β·ln(\|W_l·x\| + δ)`                              |
| `EML-noexp` | ablation | EML sans la branche exp                                           |
| `EML-nolog` | ablation | EML sans la branche log                                           |
| `exp-only`  | baseline | `α·exp(W·x + b)`                                                  |
| `log-only`  | baseline | `α·ln(\|W·x + b\| + δ)`                                           |
| `ReLU-MLP`  | baseline | `max(0, W·x + b)` — référence iso-paramétrique                    |
| `KAN`       | baseline | spline linéaire apprise par connexion (KAN simplifié, k = 8)      |
| `EML-v2`    | upgrade  | EML + biais dans chaque branche + log lisse `½·ln(z² + 1)`        |
| `EML→ReLU`  | upgrade  | hybride : 1re couche EML-v2, couches suivantes ReLU               |

## Cibles

| Famille       | Nom              | f                                    | Domaine (train → test)      |
|---------------|------------------|--------------------------------------|-----------------------------|
| in-domain     | `explog_product` | `exp(−x²)·ln(1 + y²)`                | [−2, 2]²                    |
| in-domain     | `power_function` | `(y + 1)^x = exp(x·ln(1 + y))`       | [0.1, 2]²                   |
| in-domain     | `logistic_log`   | `σ(x)·exp(y)·ln(z + 2)`              | [−2, 2]³                    |
| in-domain     | `cross_entropy`  | `H(σ(x)) + H(σ(y))`                  | [−3, 3]²                    |
| log-essential | `log_ratio`      | `ln(x₁/x₂)`                          | [10⁻², 10²]² log-uniforme   |
| log-essential | `product_ratio`  | `x₁·x₂/x₃ = exp(Σ ± ln xᵢ)`          | [0.5, 2]³ log-uniforme      |
| log-essential | `power_law`      | `x₁^1.5 · x₂^−0.5`                   | [0.2, 5]² log-uniforme      |
| extrapolation | `extrap_log_ratio` | `ln(x₁/x₂)`                        | [0.1, 10]² → [10, 100]²     |
| extrapolation | `extrap_power`   | `(y + 1)^x`                          | [0.1, 1]² → [1, 2]²         |
| extrapolation | `extrap_product` | `x₁·x₂/x₃`                           | [0.5, 2]³ → [2, 4]³         |
| extrapolation | `extrap_decay`   | `exp(−λt)`                           | t ∈ [0, 2.5] → [2.5, 5]     |
| radioactive   | `radioactive_decay` | `exp(−λt) + N(0, 0.05²)`, X = [t, λ] | t∈[0,5], λ∈[0.2,2]      |
| ood (contrôle) | `polynomial`    | `x³ + 2x²y − 0.5y³ + 0.3xy`          | [−2, 2]²                    |
| ood (contrôle) | `trigonometric` | `sin(πx)·cos(πy) + 0.5·sin(2πx)`     | [−1, 1]²                    |
| ood (contrôle) | `piecewise`     | `\|x\|·sign(y) + clamp(xy, ±1)`     | [−2, 2]²                    |
| extrapolation-control | `extrap_polynomial` | polynôme ci-dessus        | [−1, 1]² → [1, 2]²          |

Les cibles **log-essentielles** corrigent une faiblesse des cibles in-domain :
sur un petit intervalle, `ln(1 + y²)` ou `ln(z + 2)` ne sont que des courbes
lisses que n'importe quel réseau approche sans avoir besoin d'un log. Sur
plusieurs ordres de grandeur, ou quand la fonction est un produit de
puissances, un réseau sans log doit approcher `ln` par morceaux.

L'**extrapolation** est le test le plus exigeant : on évalue hors du domaine
d'entraînement. Un ReLU prolonge en ligne droite. Seul un réseau qui a
réellement capturé la loi exp/log peut continuer juste. Le polynôme extrapolé
sert de contrôle : si EML extrapolait bien partout, ce ne serait pas grâce à
la structure exp/log.

## Verdict global (règle fixée avant les runs)

Hypothèse H : *une couche contenant exp ET log en dur apprend mieux, à
paramètres égaux, les fonctions à structure exp/log.*

Une cible est **gagnée** si au moins un membre de la famille EML (`EML`,
`EML-v2`, `EML→ReLU`) est significativement meilleur que **chacun** des
comparateurs : `ReLU-MLP`, `KAN`, `exp-only`, `log-only`, et aussi les
ablations `EML-nolog` et `EML-noexp`. Gagner grâce à la seule exponentielle ne
compte donc pas. Toutes les p-values (membres × comparateurs × cibles) sont
corrigées ensemble par Benjamini-Hochberg.

| Verdict          | Condition                                                                   |
|------------------|-----------------------------------------------------------------------------|
| **réfutée**      | moins de 25 % des 12 cibles favorables gagnées                              |
| **soutenue**     | au moins 50 % des favorables gagnées, et taux supérieur à celui des contrôles |
| non concluante   | entre les deux                                                              |

Le critère est volontairement indulgent pour EML : il suffit qu'un seul de
ses trois membres gagne. Un échec est donc une réfutation forte. Les seuils
sont dans `stats_analysis.py` (`REFUTE_BELOW`, `SUPPORT_FROM`) et ne doivent
pas être modifiés après avoir vu les résultats.

## Statistiques

Pour chaque cible, EML est comparé à chaque autre variant par un test de
Mann-Whitney **bilatéral** (non apparié : deux architectures différentes à
même seed n'ont pas d'initialisation commune). Les p-values sont corrigées par
Benjamini-Hochberg sur toute la famille. La taille d'effet est le
rank-biserial r ∈ [−1, 1] (r > 0 ⇔ EML meilleur) et le rapport des médianes en
log10. Les runs divergents comptent comme NMSE = +inf (pire rang), ils ne sont
pas exclus. Les médianes ont un IC95 bootstrap (10 000 tirages).

---

## Installation et utilisation

```bash
pip install -r requirements.txt
```

```bash
python main.py --quick
```

`--quick` : 5 seeds, sans sweep HP ni ablation δ (≈ 15 min sur 4 cœurs CPU).

```bash
python main.py
```

Protocole complet : 25 seeds + sweeps.

```bash
python main.py --variants EML EML-v2 ReLU-MLP --targets logistic_log power_function --seeds 10
```

```bash
python main.py --analyze-only
```

Refait stats et figures à partir de `results/runs.jsonl` sans réentraîner.

```bash
python -m pytest -q
```

Options utiles : `--epochs`, `--workers` (0 = tous les CPU), `--stages`,
`--out` (dossier de sortie, défaut `results/`). Un run interrompu reprend où il
s'était arrêté : chaque run terminé est écrit immédiatement dans `runs.jsonl`.

Sorties dans `results/` : `runs.jsonl` (tous les runs bruts, y compris courbes
de loss et gradients), `RESULTS.md`, `results_summary.json`, et
`figures/fig1…fig7.pdf` + `table1_full.tex`.

---

## Implémenter un upgrade

Un variant n'est qu'une fonction qui fabrique **une couche cachée**. Pour en
ajouter un, on l'enregistre dans `primitives.py` :

```python
@register("EML-gated", "upgrade", "σ(g)·exp-branch + (1−σ(g))·log-branch")
def _eml_gated(d_in, d_out, layer_idx, cfg):
    return MyGatedEMLLayer(d_in, d_out, delta=cfg.delta)   # n'importe quel nn.Module
```

C'est tout. Le variant est automatiquement :

1. dimensionné en iso-paramétrie (la largeur est cherchée en comptant les vrais
   paramètres du modèle construit, pas une formule à la main) ;
2. entraîné sur toutes les cibles par `main.py` (ou seul avec `--variants`) ;
3. testé statistiquement contre EML et présent dans les figures, la table et
   `RESULTS.md` ;
4. couvert par les tests génériques (forme, sortie finie, iso-paramétrie à
   10 %). Pour une couche, faire hériter la classe de `Primitive` pour que les
   hooks de gradient la voient.

`layer_idx` permet les architectures mixtes (cf. `EML→ReLU`). Les
hyperparamètres d'entraînement passent par `TrainConfig` (`config.py`) : ajouter
un champ suffit pour qu'il soit sérialisé avec chaque run et qu'il distingue
les runs en cache.

Pistes d'upgrade déjà motivées par les résultats et les gradients :

- **Branche log bornée** (fait dans `EML-v2`) : avec `ln(|z| + δ)` et δ = 1e-4,
  `∂/∂z` monte à 1e4 près de z = 0, ce qui fait clipper une grosse fraction des
  steps. `½·ln(z² + 1)` est lisse, de dérivée ≤ ½, et se comporte comme `ln|z|`
  pour |z| grand.
- **Biais dans les branches** (fait dans `EML-v2`) : sans `b_l`, la singularité
  du log est collée à l'hyperplan `W_l·x = 0`, qui passe toujours par
  l'origine des données standardisées.
- **EML en première couche seulement** (`EML→ReLU`) : la structure exp/log sert
  surtout à « linéariser » l'entrée ; les couches suivantes peuvent rester
  standard.
- À explorer : gate appris entre les deux branches, initialisation de α, β à
  petite échelle, normalisation par branche, cibles en dimension plus élevée.

---

## Corrections par rapport à la version précédente

L'ancienne version avait plusieurs défauts qui invalidaient une partie des
conclusions publiées dans ce README.

| Problème | Effet | Correction |
|----------|-------|------------|
| La couche de **sortie** utilisait la primitive du variant (ReLU en sortie de ReLU-MLP) | ReLU-MLP ne pouvait pas prédire de valeur < 0 → MSE 9.4 sur `polynomial`, 1.0 sur `piecewise`, 0.63 sur `logistic_log` (dont la cible est négative quand z < −1). Toutes les victoires d'EML sur ReLU-MLP venaient de ce bug | Sortie `Linear` pour tous les variants |
| Fig. 3 **simulait** un histogramme log-normal à partir de la moyenne/écart-type | La figure ne montrait pas les vrais gradients | Échantillons réels enregistrés et tracés |
| Fig. 4 cherchait des clés qui n'existaient pas | Heatmap HP entièrement vide (NaN) | Hyperparamètres stockés dans chaque run |
| Test Mann-Whitney **unilatéral** (H₁ : EML < baseline) | Impossible de conclure qu'EML est pire, alors qu'il l'était sur 3/4 cibles | Test bilatéral + verdict better/worse/n.s. |
| d de Cohen sur des MSE | Valeurs absurdes (d = 32) sur une distribution très asymétrique | Rank-biserial + log10 du ratio des médianes |
| Runs divergents exclus des médianes et des tests | Biais du survivant en faveur des variants instables | Comptés comme +inf |
| Ablations EML-noexp/nolog = branche gelée | ~835 paramètres entraînables contre ~1 450 : comparaison non iso-paramétrique | Branche absente + largeur ré-ajustée |
| Pas de standardisation, seuil de convergence absolu 1e-3 | MSE incomparables entre cibles, « conv_rate » dépendant de l'échelle de y | X, y standardisés, NMSE |
| `%` non échappés dans `table1_full.tex` | Le reste de chaque ligne était commenté en LaTeX | Échappés |
| Le README affirmait qu'EML « domine » et « perd correctement en OOD » | Contredit par le JSON lui-même (EML perdait contre ReLU, exp-only et KAN sur 3/4 cibles in-domain) | Rapport généré automatiquement |
| `recap.py` : copie intégrale de tout le code (1 300 lignes) | Deux versions qui divergent | Supprimé |
| Exécution séquentielle, aucun résultat brut sauvegardé | ~1 h pour `--quick`, stats impossibles à refaire | Exécution parallèle, `runs.jsonl`, reprise, `--analyze-only` |

---

## Limitations

- Cibles en 2D/3D, 2 000 points : rien ne dit que les conclusions tiennent en
  haute dimension.
- KAN simplifié (splines linéaires, grille fixe) : borne inférieure de ce que
  ferait un vrai KAN.
- Un seul jeu d'hyperparamètres d'entraînement pour tous les variants ; le
  sweep ne porte que sur EML, ce qui l'avantage plutôt.
- `--quick` (5 seeds) donne une puissance statistique faible ; les chiffres de
  référence demandent le protocole complet.

## Licence

MIT.
